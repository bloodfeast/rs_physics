//! **A loop of two joints and a contact, solved as one constraint.**
//!
//! A self-contact between two bodies that share a jointed neighbour closes a loop: the
//! contact pushes the pair apart, and the two joints hold each of them to the neighbour.
//! The positional pass is coloured Gauss-Seidel, and Gauss-Seidel round a loop converges
//! slowly: the contact pushes, the joints pull back, and whatever is left when the passes
//! run out is read back as velocity. Worse, the contact's share of an *inherited* overlap
//! is charged as free, while the joint repair that undoes it is charged as moving, so the
//! loop hands the bodies momentum every step. See the module header, under every landing
//! pose.
//!
//! A block is every such loop, merged where loops share a body: the bodies, every joint
//! with both ends among them and every pair contact with both ends among them. Its rows are
//! linearised from where the bodies are when the pass reaches it, and solved together: a
//! dense `J M^-1 J^T` factored once, with the unilateral rows (contact normals, cones and
//! hinge ranges that are violated) handled by a drop-only active set. The moving and free
//! shares are solved separately on the same factor, so the free share removes inherited
//! overlap *while keeping the joints*, and nothing is left for a joint to pull back.
//!
//! Blocks share no body, by construction, so every block is one stage and a lane takes a
//! slice of them, exactly as it takes a slice of a colour. Each block is solved wholly by
//! one lane in a fixed order, so the answer is the same at any lane count.

use super::contacts::{surfaces, Surfaces};
use super::*;

/// One row of a block: the bodies it names, by local index, its Jacobian on each, and what
/// it asks for, split into the share that may read back as velocity and the share that may
/// not.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Row {
    a: u32,
    b: u32,
    linear_a: (f64, f64, f64),
    angular_a: (f64, f64, f64),
    linear_b: (f64, f64, f64),
    angular_b: (f64, f64, f64),
    moving: f64,
    free: f64,
    /// A row that may only push: a contact normal, or a range that is being exceeded.
    unilateral: bool,
    /// The contact this row is the normal of, or `u32::MAX`.
    contact: u32,
}

/// One block body's state for the length of one solve: where it is, and what the rows
/// have asked of it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Local {
    gathered: Gathered,
    linear: (f64, f64, f64),
    angular: (f64, f64, f64),
    free_linear: (f64, f64, f64),
    free_angular: (f64, f64, f64),
}

impl Default for Local {
    fn default() -> Self {
        let pose = Pose {
            position: (0.0, 0.0, 0.0),
            orientation: Quaternion::identity(),
            inv_mass: 0.0,
            inv_inertia: (0.0, 0.0, 0.0),
            world_inv_inertia: SymMat3::of(Quaternion::identity(), (0.0, 0.0, 0.0)),
        };
        Local {
            gathered: Gathered {
                now: pose,
                prev_position: (0.0, 0.0, 0.0),
                prev_orientation: Quaternion::identity(),
                radius: 0.0,
            },
            linear: (0.0, 0.0, 0.0),
            angular: (0.0, 0.0, 0.0),
            free_linear: (0.0, 0.0, 0.0),
            free_angular: (0.0, 0.0, 0.0),
        }
    }
}

/// Every block this step, flattened, plus the scratch each one solves in.
#[derive(Clone, Debug, Default)]
pub(super) struct Blocks {
    /// Block `k`'s bodies are `bodies[body_start[k]..body_start[k + 1]]`, sorted, and
    /// likewise its joints and its contacts.
    pub bodies: Vec<u32>,
    pub body_start: Vec<u32>,
    pub joints: Vec<u32>,
    pub joint_start: Vec<u32>,
    pub contacts: Vec<u32>,
    pub contact_start: Vec<u32>,
    /// Where block `k`'s rows and its matrix scratch begin. Sized for the most rows the
    /// block can have, which is every row of every joint and every contact.
    pub row_start: Vec<u32>,
    pub matrix_start: Vec<usize>,
    /// How many of the blocks, from the front, have a foreground body.
    pub near: usize,
    /// Per contact and per joint: solved by a block this step, so the coloured stages
    /// leave its positional normal, or the whole joint, alone.
    pub contact_in: Vec<bool>,
    pub joint_in: Vec<bool>,
    /// Scratch, reused. `rows`, `matrix` and `locals` are cut into one region per block.
    pub rows: Vec<Row>,
    pub matrix: Vec<f64>,
    pub locals: Vec<Local>,
    parent: Vec<u32>,
    touched: Vec<u32>,
    order: Vec<(u32, u32)>,
    block_of: Vec<u32>,
    tally: Vec<u32>,
    flags: Vec<bool>,
}

const NONE: u32 = u32::MAX;

impl Blocks {
    pub fn len(&self) -> usize {
        self.body_start.len().saturating_sub(1)
    }

    fn root(&mut self, mut i: u32) -> u32 {
        while self.parent[i as usize] != i {
            let up = self.parent[self.parent[i as usize] as usize];
            self.parent[i as usize] = up;
            i = up;
        }
        i
    }

    fn enter(&mut self, i: u32) {
        if self.parent[i as usize] == NONE {
            self.parent[i as usize] = i;
            self.touched.push(i);
        }
    }

    /// The lower root wins, so a block's root is its lowest body whatever order the loops
    /// were found in.
    fn union(&mut self, a: u32, b: u32) {
        self.enter(a);
        self.enter(b);
        let (ra, rb) = (self.root(a), self.root(b));
        if ra != rb {
            let (low, high) = (ra.min(rb), ra.max(rb));
            self.parent[high as usize] = low;
        }
    }

    /// **Which loops there are this step, and the blocks they make.**
    ///
    /// A pair contact whose two bodies share a jointed neighbour is a loop; its three bodies
    /// are joined, and loops that share a body are one block. Then every joint and every pair
    /// contact with both ends in one block belongs to it. A step with no such contact makes no
    /// block and costs one walk of the contact list.
    #[allow(clippy::too_many_arguments)]
    pub fn find(
        &mut self,
        bodies: usize,
        contacts: &[Contact],
        joints: &[Joint],
        jointed: Jointed<'_>,
        jointed_joint: &[u32],
        background: &BitSet,
        any_background: bool,
    ) {
        self.bodies.clear();
        self.body_start.clear();
        self.joints.clear();
        self.joint_start.clear();
        self.contacts.clear();
        self.contact_start.clear();
        self.row_start.clear();
        self.matrix_start.clear();
        self.near = 0;
        self.contact_in.clear();
        self.contact_in.resize(contacts.len(), false);
        self.joint_in.clear();
        self.joint_in.resize(joints.len(), false);
        if self.parent.len() < bodies {
            self.parent.resize(bodies, NONE);
            self.block_of.resize(bodies, NONE);
        }

        for contact in contacts.iter() {
            let (a, b) = (contact.a, contact.b);
            for &c in jointed.of(a) {
                if jointed.holds(b, c as usize) {
                    self.union(a as u32, c);
                    self.union(b as u32, c);
                }
            }
        }
        if self.touched.is_empty() {
            return;
        }

        // Group the touched bodies by root, roots ascending and bodies ascending within.
        let mut order = std::mem::take(&mut self.order);
        order.clear();
        let touched = std::mem::take(&mut self.touched);
        for &i in touched.iter() {
            let root = self.root(i);
            order.push((root, i));
        }
        order.sort_unstable();
        let mut count = 0u32;
        let mut last = NONE;
        for &(root, i) in order.iter() {
            if root != last {
                last = root;
                count += 1;
            }
            self.block_of[i as usize] = count - 1;
        }
        let blocks = count as usize;

        // Foreground blocks first, so the reduced pass takes a prefix. A block is
        // background only where every body in it is.
        let flags = &mut self.flags;
        flags.clear();
        flags.resize(blocks, true);
        if any_background {
            for &(_, i) in order.iter() {
                if !background.get(i as usize) {
                    flags[self.block_of[i as usize] as usize] = false;
                }
            }
        } else {
            flags.iter_mut().for_each(|f| *f = false);
        }
        // `tally[k]` becomes block `k`'s position in the final order.
        self.tally.clear();
        self.tally.resize(blocks, 0);
        let mut next = 0u32;
        for pass in [false, true] {
            for k in 0..blocks {
                if self.flags[k] == pass {
                    self.tally[k] = next;
                    next += 1;
                }
            }
            if !pass {
                self.near = next as usize;
            }
        }
        for &(_, i) in order.iter() {
            let k = self.block_of[i as usize];
            self.block_of[i as usize] = self.tally[k as usize];
        }

        // Bodies, by final block then by body.
        for entry in order.iter_mut() {
            entry.0 = self.block_of[entry.1 as usize];
        }
        order.sort_unstable();
        self.body_start.push(0);
        let mut current = 0u32;
        for &(k, i) in order.iter() {
            while current < k {
                self.body_start.push(self.bodies.len() as u32);
                current += 1;
            }
            self.bodies.push(i);
        }
        self.body_start.push(self.bodies.len() as u32);

        // Joints with both ends in one block, block by block, each found once from its
        // lower end.
        self.joint_start.push(0);
        for k in 0..blocks {
            let from = self.body_start[k] as usize;
            let to = self.body_start[k + 1] as usize;
            let begin = self.joints.len();
            for n in from..to {
                let i = self.bodies[n] as usize;
                let run = jointed.start[i] as usize..jointed.start[i + 1] as usize;
                for at in run {
                    let other = jointed.to[at] as usize;
                    if other > i && self.block_of[other] == k as u32 {
                        let j = jointed_joint[at];
                        self.joints.push(j);
                        self.joint_in[j as usize] = true;
                    }
                }
            }
            self.joints[begin..].sort_unstable();
            self.joint_start.push(self.joints.len() as u32);
        }

        // Contacts with both ends in one block, by a counting sort on the block so each
        // block's list stays in contact order.
        self.tally.clear();
        self.tally.resize(blocks + 1, 0);
        for (index, contact) in contacts.iter().enumerate() {
            let k = self.block_of[contact.a];
            if k != NONE && k == self.block_of[contact.b] {
                self.tally[k as usize + 1] += 1;
                self.contact_in[index] = true;
            }
        }
        for k in 0..blocks {
            self.tally[k + 1] += self.tally[k];
        }
        self.contact_start.extend_from_slice(&self.tally);
        self.contacts.resize(self.tally[blocks] as usize, 0);
        for (index, contact) in contacts.iter().enumerate() {
            if self.contact_in[index] {
                let k = self.block_of[contact.a] as usize;
                self.contacts[self.tally[k] as usize] = index as u32;
                self.tally[k] += 1;
            }
        }

        // Scratch: the most rows each block can carry, and a matrix region for them.
        self.row_start.push(0);
        self.matrix_start.push(0);
        let mut rows_total = 0usize;
        let mut matrix_total = 0usize;
        for k in 0..blocks {
            let mut rows = 0usize;
            for &j in &self.joints[self.joint_start[k] as usize..self.joint_start[k + 1] as usize]
            {
                rows += match joints[j as usize] {
                    Joint::Ball { cone, .. } => 3 + usize::from(cone < std::f64::consts::PI),
                    Joint::Hinge { .. } => 6,
                };
            }
            rows += (self.contact_start[k + 1] - self.contact_start[k]) as usize;
            rows_total += rows;
            matrix_total += scratch_for(rows);
            self.row_start.push(rows_total as u32);
            self.matrix_start.push(matrix_total);
        }
        if self.rows.len() < rows_total {
            self.rows.resize(rows_total, Row::default());
        }
        if self.matrix.len() < matrix_total {
            self.matrix.resize(matrix_total, 0.0);
        }
        if self.locals.len() < self.bodies.len() {
            self.locals.resize(self.bodies.len(), Local::default());
        }

        for &(_, i) in order.iter() {
            self.parent[i as usize] = NONE;
            self.block_of[i as usize] = NONE;
        }
        self.order = order;
        let mut touched = touched;
        touched.clear();
        self.touched = touched;
    }
}

/// The matrix scratch a block of `n` rows needs: the matrix, its factor, and five vectors.
fn scratch_for(n: usize) -> usize {
    2 * n * n + 5 * n
}

/// What one block solve reads, bundled so the stage hands over one value.
pub(super) struct View<'a> {
    pub bodies: &'a [u32],
    pub body_start: &'a [u32],
    pub joints: &'a [u32],
    pub joint_start: &'a [u32],
    pub contacts: &'a [u32],
    pub contact_start: &'a [u32],
    pub row_start: &'a [u32],
    pub matrix_start: &'a [usize],
    pub rows: scatter::Cells<'a, Row>,
    pub matrix: scatter::Cells<'a, f64>,
    pub locals: scatter::Cells<'a, Local>,
}

/// **One block, one pass**: linearise every row at the poses the pass has reached, solve
/// them together, and apply the result.
///
/// # Safety
/// The block's bodies, its contacts' running impulses and its scratch regions must be
/// owned by this thread for the call, which they are: blocks share no body, a contact is
/// in at most one block, and each block's scratch is its own region.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn solve(
    k: usize,
    view: &View<'_>,
    joints: &[Joint],
    contacts: &[Contact],
    bodies: &Bodies,
    contact_impulse: &scatter::Cells<Spent>,
    inv_mass: &[f64],
    inv_inertia: &[(f64, f64, f64)],
    radius: &[f64],
) {
    let members = &view.bodies[view.body_start[k] as usize..view.body_start[k + 1] as usize];
    let locals = view
        .locals
        .region(view.body_start[k] as usize..view.body_start[k + 1] as usize);
    for (local, &i) in locals.iter_mut().zip(members.iter()) {
        *local = Local {
            gathered: bodies.gather(i as usize, inv_mass, inv_inertia, radius),
            ..Local::default()
        };
    }
    let local_of = |i: usize| members.binary_search(&(i as u32)).unwrap_or(0) as u32;

    let capacity = (view.row_start[k + 1] - view.row_start[k]) as usize;
    let all_rows = view
        .rows
        .region(view.row_start[k] as usize..view.row_start[k + 1] as usize);
    let mut n = 0usize;
    let mut push = |row: Row| {
        all_rows[n] = row;
        n += 1;
    };

    // -- the joints' rows, bilateral first --------------------------------------
    for &j in &view.joints[view.joint_start[k] as usize..view.joint_start[k + 1] as usize] {
        let joint = joints[j as usize];
        let (a, b) = joint.bodies();
        let (la, lb) = (local_of(a), local_of(b));
        let first = locals[la as usize].gathered.now;
        let second = locals[lb as usize].gathered.now;
        let (anchor_a, anchor_b) = match joint {
            Joint::Ball {
                anchor_a, anchor_b, ..
            } => (anchor_a, anchor_b),
            Joint::Hinge {
                anchor_a, anchor_b, ..
            } => (anchor_a, anchor_b),
        };
        let ra = rotate(first.orientation, anchor_a);
        let rb = rotate(second.orientation, anchor_b);
        let error = sub(add(second.position, rb), add(first.position, ra));
        // **The point as three rows, not one along the error.** One row along the error has
        // no direction when the joint is satisfied, so the block would not see the joint
        // resist a contact at all; the three rows are the same constraint in vector form.
        for e in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)] {
            push(Row {
                a: la,
                b: lb,
                linear_a: scale(e, -1.0),
                angular_a: scale(cross(ra, e), -1.0),
                linear_b: e,
                angular_b: cross(rb, e),
                moving: -dot(error, e),
                free: 0.0,
                unilateral: false,
                contact: NONE,
            });
        }
        let turn = |push: &mut dyn FnMut(Row), axis: (f64, f64, f64), wanted: f64, one_sided| {
            push(Row {
                a: la,
                b: lb,
                linear_a: (0.0, 0.0, 0.0),
                angular_a: axis,
                linear_b: (0.0, 0.0, 0.0),
                angular_b: scale(axis, -1.0),
                moving: wanted,
                free: 0.0,
                unilateral: one_sided,
                contact: NONE,
            })
        };
        match joint {
            Joint::Ball {
                axis_a,
                axis_b,
                cone,
                ..
            } if cone < std::f64::consts::PI => {
                // The cone as solve_joint states it: past `cone`, turn the pair together
                // about `a x b` by the excess, and never the other way.
                let world_a = rotate(first.orientation, axis_a);
                let world_b = rotate(second.orientation, axis_b);
                let across = cross(world_a, world_b);
                let along = dot(world_a, world_b);
                let acute = length(across).clamp(-1.0, 1.0).asin();
                let angle = if along >= 0.0 {
                    acute
                } else {
                    std::f64::consts::PI - acute
                };
                let excess = angle - cone;
                if excess > 0.0 {
                    let axis = match normalized(across) {
                        Some(n) => Some(n),
                        None if along < 0.0 => Some(perpendicular(world_a)),
                        None => None,
                    };
                    if let Some(axis) = axis {
                        turn(&mut push, axis, excess, true);
                    }
                }
            }
            Joint::Hinge {
                axis_a,
                axis_b,
                min,
                max,
                ..
            } => {
                // The alignment as two rows about the directions perpendicular to the hinge,
                // asking for the turn solve_joint asks for: `angle` about `a x b`.
                let world_a = rotate(first.orientation, axis_a);
                let world_b = rotate(second.orientation, axis_b);
                let across = cross(world_a, world_b);
                let along = dot(world_a, world_b);
                let acute = length(across).clamp(-1.0, 1.0).asin();
                let angle = if along >= 0.0 {
                    acute
                } else {
                    std::f64::consts::PI - acute
                };
                let wanted = match normalized(across) {
                    Some(n) => scale(n, angle),
                    None if along < 0.0 => scale(perpendicular(world_a), std::f64::consts::PI),
                    None => (0.0, 0.0, 0.0),
                };
                if let Some(axis) = normalized(world_a) {
                    let t1 = perpendicular(axis);
                    let t2 = cross(axis, t1);
                    turn(&mut push, t1, dot(wanted, t1), false);
                    turn(&mut push, t2, dot(wanted, t2), false);
                    let angle = hinge_angle(
                        first.orientation,
                        second.orientation,
                        axis,
                        axis_a,
                        axis_b,
                    );
                    if angle > max {
                        turn(&mut push, axis, angle - max, true);
                    } else if angle < min {
                        turn(&mut push, scale(axis, -1.0), min - angle, true);
                    }
                }
            }
            Joint::Ball { .. } => {}
        }
    }

    // -- the contacts' normal rows, where the pair overlaps now ------------------
    for &c in &view.contacts[view.contact_start[k] as usize..view.contact_start[k + 1] as usize] {
        let contact = contacts[c as usize];
        let (la, lb) = (local_of(contact.a), local_of(contact.b));
        let first = locals[la as usize].gathered;
        let second = locals[lb as usize].gathered;
        let Surfaces {
            ra,
            rb,
            depth,
            slid,
        } = surfaces(&contact, &first, &second);
        if depth <= 0.0 {
            continue;
        }
        let spent = contact_impulse.get(c as usize);
        // Exactly the split solve_contact_normal makes: what the step drove may read back
        // as velocity, what it inherited may not, and a revived pair inherited all of it.
        let driven = if contact.revived && spent.normal > 0.0 {
            0.0
        } else {
            dot(slid, contact.normal).clamp(0.0, depth)
        };
        let normal = contact.normal;
        push(Row {
            a: la,
            b: lb,
            linear_a: scale(normal, -1.0),
            angular_a: scale(cross(ra, normal), -1.0),
            linear_b: normal,
            angular_b: cross(rb, normal),
            moving: driven,
            free: depth - driven,
            unilateral: true,
            contact: c,
        });
    }
    debug_assert!(n <= capacity);
    if n == 0 {
        return;
    }
    let rows = &all_rows[..n];

    // -- J M^-1 J^T ------------------------------------------------------------
    let scratch = view
        .matrix
        .region(view.matrix_start[k]..view.matrix_start[k] + scratch_for(n));
    let (matrix, rest) = scratch.split_at_mut(n * n);
    let (factor, rest) = rest.split_at_mut(n * n);
    let (lambda, rest) = rest.split_at_mut(n);
    let (moving, rest) = rest.split_at_mut(n);
    let (free, rest) = rest.split_at_mut(n);
    let (rhs, rest) = rest.split_at_mut(n);
    let active = &mut rest[..n];

    let through = |body: &Pose, linear: (f64, f64, f64), angular: (f64, f64, f64), l2, a2| {
        body.inv_mass * dot(linear, l2) + dot(angular, body.world_inv_inertia.apply(a2))
    };
    for r in 0..n {
        for s in r..n {
            let (x, y) = (rows[r], rows[s]);
            let mut sum = 0.0;
            for (body, l1, a1) in [(x.a, x.linear_a, x.angular_a), (x.b, x.linear_b, x.angular_b)]
            {
                let pose = &locals[body as usize].gathered.now;
                if y.a == body {
                    sum += through(pose, l1, a1, y.linear_a, y.angular_a);
                }
                if y.b == body {
                    sum += through(pose, l1, a1, y.linear_b, y.angular_b);
                }
            }
            matrix[r * n + s] = sum;
            matrix[s * n + r] = sum;
        }
    }

    // -- the drop-only active set ---------------------------------------------
    //
    // Every row starts active. A unilateral row the solve asks to pull is dropped, the most
    // negative first, and the rest solved again; each drop removes a row for good, so this
    // ends in at most one solve per unilateral row.
    for r in 0..n {
        active[r] = 1.0;
        rhs[r] = rows[r].moving + rows[r].free;
    }
    loop {
        factorise(matrix, factor, active, n);
        substitute(factor, active, rhs, lambda, n);
        let mut worst = 0.0;
        let mut worst_at = usize::MAX;
        for r in 0..n {
            if rows[r].unilateral && active[r] > 0.0 && lambda[r] < worst {
                worst = lambda[r];
                worst_at = r;
            }
        }
        if worst_at == usize::MAX {
            break;
        }
        active[worst_at] = 0.0;
    }
    for r in 0..n {
        rhs[r] = rows[r].moving;
    }
    substitute(factor, active, rhs, moving, n);
    for r in 0..n {
        rhs[r] = rows[r].free;
    }
    substitute(factor, active, rhs, free, n);

    // -- what each body was asked for, and the contacts' running totals -------------
    for (r, row) in rows.iter().enumerate() {
        if active[r] <= 0.0 {
            continue;
        }
        for (body, linear, angular) in [
            (row.a, row.linear_a, row.angular_a),
            (row.b, row.linear_b, row.angular_b),
        ] {
            let local = &mut locals[body as usize];
            local.linear = add(local.linear, scale(linear, moving[r]));
            local.angular = add(local.angular, scale(angular, moving[r]));
            local.free_linear = add(local.free_linear, scale(linear, free[r]));
            local.free_angular = add(local.free_angular, scale(angular, free[r]));
        }
        if row.contact != NONE {
            let total = lambda[r].max(0.0);
            let mut spent = contact_impulse.get(row.contact as usize);
            spent.normal += total;
            spent.driven += moving[r].clamp(0.0, total);
            contact_impulse.set(row.contact as usize, spent);
        }
    }
    for (local, &i) in locals.iter().zip(members.iter()) {
        let body = &local.gathered.now;
        if !body.movable() {
            continue;
        }
        let turn = |angular: (f64, f64, f64)| {
            let dw = body.world_inv_inertia.apply(angular);
            renormalized(Quaternion {
                w: 1.0,
                x: 0.5 * dw.0,
                y: 0.5 * dw.1,
                z: 0.5 * dw.2,
            })
        };
        let correction = Correction {
            body: i as usize,
            translation: scale(local.linear, body.inv_mass),
            rotation: turn(local.angular),
            free_translation: scale(local.free_linear, body.inv_mass),
            free_rotation: turn(local.free_angular),
        };
        bodies.apply([correction, Correction::none()]);
    }
}

/// Cholesky of the active rows of `matrix` into `factor`, lower triangle, row-major.
///
/// **A row whose pivot falls to [`SATURATED`] of its own diagonal depends on the rows
/// before it**, and is left out of this pass: its column is zeroed and it is marked
/// inactive. The joint rows come first, so where the joints already decide a contact it is
/// the contact that is dropped -- the overlap the joints forbid removing is not one this
/// solve can remove either.
fn factorise(matrix: &[f64], factor: &mut [f64], active: &mut [f64], n: usize) {
    for r in 0..n {
        for c in 0..n {
            factor[r * n + c] = 0.0;
        }
    }
    for r in 0..n {
        if active[r] <= 0.0 {
            continue;
        }
        for c in 0..=r {
            if active[c] <= 0.0 {
                continue;
            }
            let mut sum = matrix[r * n + c];
            for m in 0..c {
                sum -= factor[r * n + m] * factor[c * n + m];
            }
            if c == r {
                let diagonal = matrix[r * n + r];
                if diagonal <= 0.0 || sum <= SATURATED * diagonal {
                    active[r] = 0.0;
                    for m in 0..r {
                        factor[r * n + m] = 0.0;
                    }
                } else {
                    factor[r * n + r] = sum.sqrt();
                }
            } else {
                factor[r * n + c] = sum / factor[c * n + c];
            }
        }
    }
}

/// `L L^T x = b` over the active rows; an inactive row's `x` is zero.
fn substitute(factor: &[f64], active: &[f64], b: &[f64], x: &mut [f64], n: usize) {
    for r in 0..n {
        if active[r] <= 0.0 {
            x[r] = 0.0;
            continue;
        }
        let mut sum = b[r];
        for c in 0..r {
            sum -= factor[r * n + c] * x[c];
        }
        x[r] = sum / factor[r * n + r];
    }
    for r in (0..n).rev() {
        if active[r] <= 0.0 {
            continue;
        }
        let mut sum = x[r];
        for c in r + 1..n {
            sum -= factor[c * n + r] * x[c];
        }
        x[r] = sum / factor[r * n + r];
    }
}
