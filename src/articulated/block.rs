//! **A whole rig, with its self-contacts and its ground, solved as one constraint.**
//!
//! While a skeleton may touch itself, every rig -- every set of bodies the live joints
//! connect -- is solved directly once a pass rather than by the coloured Gauss-Seidel
//! sweep. A coloured sweep converges slowly round a loop, and a rig lying on the ground is
//! nothing but loops: two bones touching close one through the joints between them, and
//! two ends of two bones on the plane close another through the ground. Worse, the sweep
//! charges the two halves of each fight differently: a contact's push out of an inherited
//! overlap is free, and the joints' pull back is moving, so a loop hands the bodies
//! momentum every step. See the module header, under every landing pose.
//!
//! **The rows.** Every joint's point as three rows and a hinge's alignment as two -- the
//! bilateral *head*, solved first -- and then the one-sided *tail*: a hinge range or a cone
//! that is being exceeded, both ends of the ground patch under every body, and every pair
//! contact with both bodies in the rig. All are linearised from where the bodies are when
//! the pass reaches the rig.
//!
//! **The solve.** `K = J M^-1 J^T`, dense, factored with the head first, so the tail is
//! solved on the Schur complement the head leaves. The tail is a complementarity problem
//! -- a contact may push and not pull, and may not be left inside what it touches -- and
//! is solved by an active set: drop the most negative, and once nothing pulls, re-admit
//! the row left deepest, until neither applies or the re-solves reach the row count. A row
//! whose pivot falls to [`SATURATED`] of its own diagonal depends on the rows before it
//! and is left out of the pass.
//!
//! **The charge.** The share of each overlap the step drove may read back as velocity and
//! the share it inherited may not. The two are solved separately on the same factor, so
//! the inherited share is removed *while keeping the joints*, and nothing is left for a
//! joint to pull back as momentum.
//!
//! Rigs share no body, so every rig is one stage and a lane takes a slice of them, each
//! solved wholly by one lane in a fixed order with scratch the lane's thread owns; the
//! answer is the same at any lane count.

use std::cell::RefCell;

use super::contacts::{surfaces, GroundContact, Patch, Surfaces};
use super::*;

/// What a row is the row of, which decides what its answer is written back to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Kind {
    #[default]
    Joint,
    /// A range or a cone being exceeded.
    Limit,
    /// One end of a ground patch: the patch, and which end.
    Ground { patch: u32, end: u8 },
    /// A pair contact.
    Pair { contact: u32 },
}

/// One row: the bodies it names by local index (`b` is [`NONE`] against the ground), its
/// Jacobian on each, the same through each body's inverse mass, and what it asks for,
/// split into the share that may read back as velocity and the share that may not.
#[derive(Clone, Copy, Debug, Default)]
struct Row {
    a: u32,
    b: u32,
    linear_a: (f64, f64, f64),
    angular_a: (f64, f64, f64),
    linear_b: (f64, f64, f64),
    angular_b: (f64, f64, f64),
    through_linear_a: (f64, f64, f64),
    through_angular_a: (f64, f64, f64),
    through_linear_b: (f64, f64, f64),
    through_angular_b: (f64, f64, f64),
    moving: f64,
    free: f64,
    kind: Kind,
    /// Whether the active set starts with it: a one-sided row that is not yet violated
    /// starts out and comes in only if the solve drives it in.
    starts_active: bool,
}

/// One rig body's state for the length of one solve.
#[derive(Clone, Copy, Debug)]
struct Local {
    gathered: Gathered,
    linear: (f64, f64, f64),
    angular: (f64, f64, f64),
    free_linear: (f64, f64, f64),
    free_angular: (f64, f64, f64),
}

const ACTIVE: u8 = 0;
const OUT: u8 = 1;
/// Left out because it depends on the rows before it. Not re-admitted in the same pass.
const DEPENDENT: u8 = 2;

/// Everything one rig solve writes, owned by the thread that runs it and reused.
#[derive(Default)]
struct Scratch {
    locals: Vec<Local>,
    rows: Vec<Row>,
    limits: Vec<Row>,
    matrix: Vec<f64>,
    factor: Vec<f64>,
    schur: Vec<f64>,
    lambda: Vec<f64>,
    moving: Vec<f64>,
    free: Vec<f64>,
    rhs: Vec<f64>,
    work: Vec<f64>,
    state: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Every rig this step, flattened.
#[derive(Clone, Debug, Default)]
pub(super) struct Blocks {
    /// Rig `k`'s bodies are `bodies[body_start[k]..body_start[k + 1]]`, sorted, and
    /// likewise its joints, its pair contacts and its ground patches, each in index order.
    pub bodies: Vec<u32>,
    pub body_start: Vec<u32>,
    pub joints: Vec<u32>,
    pub joint_start: Vec<u32>,
    pub contacts: Vec<u32>,
    pub contact_start: Vec<u32>,
    pub grounds: Vec<u32>,
    pub ground_start: Vec<u32>,
    /// How many of the rigs, from the front, have a foreground body.
    pub near: usize,
    /// Per pair contact and per ground patch: the rig that solves its normal this step,
    /// or [`NONE`]. A coloured normal stage leaves it alone when that rig is in the plan
    /// it is running.
    pub contact_block: Vec<u32>,
    pub ground_block: Vec<u32>,
    /// Per joint: solved by a rig this step, so the coloured stages leave it alone.
    pub joint_in: Vec<bool>,
    parent: Vec<u32>,
    touched: Vec<u32>,
    order: Vec<(u32, u32)>,
    block_of: Vec<u32>,
    tally: Vec<u32>,
    flags: Vec<bool>,
}

pub(super) const NONE: u32 = u32::MAX;

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

    /// The lower root wins, so a rig's root is its lowest body whatever order its joints
    /// are in.
    fn union(&mut self, a: u32, b: u32) {
        self.enter(a);
        self.enter(b);
        let (ra, rb) = (self.root(a), self.root(b));
        if ra != rb {
            let (low, high) = (ra.min(rb), ra.max(rb));
            self.parent[high as usize] = low;
        }
    }

    /// **The rigs this step.** Nothing while `self_collision` is off, so that path is
    /// exactly what it was. Otherwise every joint with an awake end joins its two bodies,
    /// and each set so joined is a rig: its joints, every pair contact with both bodies in
    /// it, and the ground patch of every body in it. A body with no live joint is in none.
    #[allow(clippy::too_many_arguments)]
    pub fn find(
        &mut self,
        bodies: usize,
        self_collision: bool,
        joints: &[Joint],
        awake: &BitSet,
        contacts: &[Contact],
        grounds: &[GroundContact],
        background: &BitSet,
        any_background: bool,
    ) {
        self.bodies.clear();
        self.body_start.clear();
        self.joints.clear();
        self.joint_start.clear();
        self.contacts.clear();
        self.contact_start.clear();
        self.grounds.clear();
        self.ground_start.clear();
        self.near = 0;
        self.contact_block.clear();
        self.contact_block.resize(contacts.len(), NONE);
        self.ground_block.clear();
        self.ground_block.resize(grounds.len(), NONE);
        self.joint_in.clear();
        self.joint_in.resize(joints.len(), false);
        if !self_collision || joints.is_empty() {
            return;
        }
        if self.parent.len() < bodies {
            self.parent.resize(bodies, NONE);
            self.block_of.resize(bodies, NONE);
        }

        let live = |joint: &Joint| {
            let (a, b) = joint.bodies();
            awake.get(a) || awake.get(b)
        };
        for joint in joints.iter() {
            if live(joint) {
                let (a, b) = joint.bodies();
                self.union(a as u32, b as u32);
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

        // Foreground rigs first, so the reduced pass takes a prefix. A rig is background
        // only where every body in it is.
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

        // Bodies, by final rig then by body.
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

        // Joints, pair contacts and ground patches, each by a counting sort on the rig so
        // every rig's list stays in index order.
        let block_of = &self.block_of;
        let tally = &mut self.tally;
        let mut sort = |of: &mut dyn FnMut(usize) -> Option<u32>,
                        len: usize,
                        list: &mut Vec<u32>,
                        start: &mut Vec<u32>,
                        mark: &mut dyn FnMut(usize, u32)| {
            tally.clear();
            tally.resize(blocks + 1, 0);
            for index in 0..len {
                if let Some(k) = of(index) {
                    tally[k as usize + 1] += 1;
                }
            }
            for k in 0..blocks {
                tally[k + 1] += tally[k];
            }
            start.extend_from_slice(tally);
            list.resize(tally[blocks] as usize, 0);
            for index in 0..len {
                if let Some(k) = of(index) {
                    list[tally[k as usize] as usize] = index as u32;
                    tally[k as usize] += 1;
                    mark(index, k);
                }
            }
        };
        let joint_in = &mut self.joint_in;
        sort(
            &mut |index| {
                let joint = &joints[index];
                if !live(joint) {
                    return None;
                }
                let k = block_of[joint.bodies().0];
                (k != NONE).then_some(k)
            },
            joints.len(),
            &mut self.joints,
            &mut self.joint_start,
            &mut |index, _| joint_in[index] = true,
        );
        let contact_block = &mut self.contact_block;
        sort(
            &mut |index| {
                let contact = &contacts[index];
                let k = block_of[contact.a];
                (k != NONE && k == block_of[contact.b]).then_some(k)
            },
            contacts.len(),
            &mut self.contacts,
            &mut self.contact_start,
            &mut |index, k| contact_block[index] = k,
        );
        let ground_block = &mut self.ground_block;
        sort(
            &mut |index| {
                let k = block_of[grounds[index].body];
                (k != NONE).then_some(k)
            },
            grounds.len(),
            &mut self.grounds,
            &mut self.ground_start,
            &mut |index, k| ground_block[index] = k,
        );

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

/// What one rig solve reads, bundled so the stage hands over one value.
pub(super) struct View<'a> {
    pub bodies: &'a [u32],
    pub body_start: &'a [u32],
    pub joints: &'a [u32],
    pub joint_start: &'a [u32],
    pub contacts: &'a [u32],
    pub contact_start: &'a [u32],
    pub grounds: &'a [u32],
    pub ground_start: &'a [u32],
}

/// **One rig, one pass**: linearise every row at the poses the pass has reached, solve
/// them together, and apply the result.
///
/// # Safety
/// The rig's bodies, its contacts' and patches' running impulses must be owned by this
/// thread for the call, which they are: rigs share no body, and a contact or a patch is in
/// at most one rig.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn solve(
    k: usize,
    view: &View<'_>,
    joints: &[Joint],
    contacts: &[Contact],
    grounds: &[GroundContact],
    bodies: &Bodies,
    contact_impulse: &scatter::Cells<Spent>,
    ground_impulse: &scatter::Cells<Patch>,
    inv_mass: &[f64],
    inv_inertia: &[(f64, f64, f64)],
    radius: &[f64],
) {
    SCRATCH.with(|scratch| {
        let scratch = &mut *scratch.borrow_mut();
        solve_in(
            scratch,
            k,
            view,
            joints,
            contacts,
            grounds,
            bodies,
            contact_impulse,
            ground_impulse,
            inv_mass,
            inv_inertia,
            radius,
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn solve_in(
    scratch: &mut Scratch,
    k: usize,
    view: &View<'_>,
    joints: &[Joint],
    contacts: &[Contact],
    grounds: &[GroundContact],
    bodies: &Bodies,
    contact_impulse: &scatter::Cells<Spent>,
    ground_impulse: &scatter::Cells<Patch>,
    inv_mass: &[f64],
    inv_inertia: &[(f64, f64, f64)],
    radius: &[f64],
) {
    let members = &view.bodies[view.body_start[k] as usize..view.body_start[k + 1] as usize];
    let Scratch {
        locals,
        rows,
        limits,
        matrix,
        factor,
        schur,
        lambda,
        moving,
        free,
        rhs,
        work,
        state,
    } = scratch;
    locals.clear();
    for &i in members.iter() {
        locals.push(Local {
            gathered: bodies.gather(i as usize, inv_mass, inv_inertia, radius),
            linear: (0.0, 0.0, 0.0),
            angular: (0.0, 0.0, 0.0),
            free_linear: (0.0, 0.0, 0.0),
            free_angular: (0.0, 0.0, 0.0),
        });
    }
    let local_of = |i: usize| members.binary_search(&(i as u32)).unwrap_or(0) as u32;
    rows.clear();
    limits.clear();

    // -- the joints: the point and the alignment in the head, ranges in the tail -------
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
        // no direction when the joint is satisfied, so the solve would not see the joint
        // resist a contact at all; the three rows are the same constraint in vector form.
        for e in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)] {
            rows.push(Row {
                a: la,
                b: lb,
                linear_a: scale(e, -1.0),
                angular_a: scale(cross(ra, e), -1.0),
                linear_b: e,
                angular_b: cross(rb, e),
                moving: -dot(error, e),
                kind: Kind::Joint,
                starts_active: true,
                ..Row::default()
            });
        }
        let turn = |axis: (f64, f64, f64), wanted: f64, kind: Kind| Row {
            a: la,
            b: lb,
            angular_a: axis,
            angular_b: scale(axis, -1.0),
            moving: wanted,
            kind,
            starts_active: true,
            ..Row::default()
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
                        limits.push(turn(axis, excess, Kind::Limit));
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
                    rows.push(turn(t1, dot(wanted, t1), Kind::Joint));
                    rows.push(turn(t2, dot(wanted, t2), Kind::Joint));
                    let angle = hinge_angle(
                        first.orientation,
                        second.orientation,
                        axis,
                        axis_a,
                        axis_b,
                    );
                    if angle > max {
                        limits.push(turn(axis, angle - max, Kind::Limit));
                    } else if angle < min {
                        limits.push(turn(scale(axis, -1.0), min - angle, Kind::Limit));
                    }
                }
            }
            Joint::Ball { .. } => {}
        }
    }
    let head = rows.len();
    rows.extend_from_slice(limits);

    // -- both ends of every ground patch -------------------------------------------------
    for &g in &view.grounds[view.ground_start[k] as usize..view.ground_start[k + 1] as usize] {
        let contact = grounds[g as usize];
        let la = local_of(contact.body);
        let body = locals[la as usize].gathered;
        let normal = contact.normal;
        for end in 0..2 {
            let local = contact.local[end];
            let arm = rotate(body.now.orientation, local);
            let at = add(body.now.position, arm);
            let depth = contact.distance - dot(normal, at);
            let was = add(body.prev_position, rotate(body.prev_orientation, local));
            // The share of this end's overlap the step drove into the plane, which is the
            // only part allowed to read back as velocity: solve_ground_normal's rule.
            let driven = (-dot(sub(at, was), normal)).clamp(0.0, depth.max(0.0));
            rows.push(Row {
                a: la,
                b: NONE,
                linear_a: normal,
                angular_a: cross(arm, normal),
                moving: driven,
                free: depth - driven,
                kind: Kind::Ground {
                    patch: g,
                    end: end as u8,
                },
                starts_active: depth > 0.0,
                ..Row::default()
            });
        }
    }

    // -- every pair contact inside the rig ----------------------------------------------
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
        let spent = contact_impulse.get(c as usize);
        // Exactly the split solve_contact_normal makes: what the step drove may read back
        // as velocity, what it inherited may not, and a revived pair inherited all of it.
        let driven = if depth <= 0.0 || (contact.revived && spent.normal > 0.0) {
            0.0
        } else {
            dot(slid, contact.normal).clamp(0.0, depth)
        };
        let normal = contact.normal;
        rows.push(Row {
            a: la,
            b: lb,
            linear_a: scale(normal, -1.0),
            angular_a: scale(cross(ra, normal), -1.0),
            linear_b: normal,
            angular_b: cross(rb, normal),
            moving: driven,
            free: depth - driven,
            kind: Kind::Pair { contact: c },
            starts_active: depth > 0.0,
            ..Row::default()
        });
    }
    let n = rows.len();
    if n == 0 {
        return;
    }
    let tail = n - head;

    // -- M^-1 J^T, then K = J M^-1 J^T ---------------------------------------------------
    for row in rows.iter_mut() {
        let pose = &locals[row.a as usize].gathered.now;
        row.through_linear_a = scale(row.linear_a, pose.inv_mass);
        row.through_angular_a = pose.world_inv_inertia.apply(row.angular_a);
        if row.b != NONE {
            let pose = &locals[row.b as usize].gathered.now;
            row.through_linear_b = scale(row.linear_b, pose.inv_mass);
            row.through_angular_b = pose.world_inv_inertia.apply(row.angular_b);
        }
    }
    matrix.clear();
    matrix.resize(n * n, 0.0);
    for r in 0..n {
        let x = rows[r];
        for s in r..n {
            let y = rows[s];
            let mut sum = 0.0;
            if x.a == y.a {
                sum += dot(x.linear_a, y.through_linear_a) + dot(x.angular_a, y.through_angular_a);
            }
            if x.b != NONE {
                if x.b == y.a {
                    sum += dot(x.linear_b, y.through_linear_a)
                        + dot(x.angular_b, y.through_angular_a);
                }
                if x.b == y.b {
                    sum += dot(x.linear_b, y.through_linear_b)
                        + dot(x.angular_b, y.through_angular_b);
                }
            }
            if y.b != NONE && x.a == y.b {
                sum += dot(x.linear_a, y.through_linear_b) + dot(x.angular_a, y.through_angular_b);
            }
            matrix[r * n + s] = sum;
            matrix[s * n + r] = sum;
        }
    }

    // -- the head, factored once; the tail's columns on it; the Schur complement ---------
    state.clear();
    state.extend(rows.iter().map(|row| if row.starts_active { ACTIVE } else { OUT }));
    factor.clear();
    factor.resize(n * n, 0.0);
    for r in 0..head {
        for c in 0..=r {
            if state[c] != ACTIVE {
                continue;
            }
            let mut sum = matrix[r * n + c];
            for m in 0..c {
                sum -= factor[r * n + m] * factor[c * n + m];
            }
            if c == r {
                let diagonal = matrix[r * n + r];
                if diagonal <= 0.0 || sum <= SATURATED * diagonal {
                    state[r] = DEPENDENT;
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
    for r in head..n {
        for c in 0..head {
            if state[c] != ACTIVE {
                continue;
            }
            let mut sum = matrix[r * n + c];
            for m in 0..c {
                sum -= factor[r * n + m] * factor[c * n + m];
            }
            factor[r * n + c] = sum / factor[c * n + c];
        }
    }
    schur.clear();
    schur.resize(tail * tail, 0.0);
    for i in 0..tail {
        let r = head + i;
        for j in 0..=i {
            let c = head + j;
            let mut sum = matrix[r * n + c];
            for m in 0..head {
                sum -= factor[r * n + m] * factor[c * n + m];
            }
            schur[i * tail + j] = sum;
            schur[j * tail + i] = sum;
        }
    }

    // -- the active set over the tail --------------------------------------------------
    rhs.clear();
    rhs.extend(rows.iter().map(|row| row.moving + row.free));
    lambda.clear();
    lambda.resize(n, 0.0);
    work.clear();
    work.resize(n, 0.0);
    let mut solves = 0usize;
    loop {
        factorise_tail(matrix, factor, schur, state, n, head);
        substitute(factor, state, rhs, lambda, work, n);
        solves += 1;
        // Anything that pulls goes, the most negative first, lowest index on ties.
        let mut worst = 0.0;
        let mut worst_at = usize::MAX;
        for r in head..n {
            if state[r] == ACTIVE && lambda[r] < worst {
                worst = lambda[r];
                worst_at = r;
            }
        }
        if worst_at != usize::MAX {
            state[worst_at] = OUT;
            continue;
        }
        if solves >= tail {
            break;
        }
        // Nothing pulls. Whatever this answer leaves deepest inside what it touches comes
        // back in.
        let mut deepest = 0.0;
        let mut deepest_at = usize::MAX;
        for r in head..n {
            if state[r] != OUT {
                continue;
            }
            let mut reached = 0.0;
            for s in 0..n {
                reached += matrix[r * n + s] * lambda[s];
            }
            let left = rhs[r] - reached;
            if left > deepest {
                deepest = left;
                deepest_at = r;
            }
        }
        if deepest_at == usize::MAX {
            break;
        }
        state[deepest_at] = ACTIVE;
    }
    tally(n, head, solves, state);

    // -- the two charges on the same factor ---------------------------------------------
    rhs.clear();
    rhs.extend(rows.iter().map(|row| row.moving));
    moving.clear();
    moving.resize(n, 0.0);
    substitute(factor, state, rhs, moving, work, n);
    rhs.clear();
    rhs.extend(rows.iter().map(|row| row.free));
    free.clear();
    free.resize(n, 0.0);
    substitute(factor, state, rhs, free, work, n);

    // -- what each body was asked for, and the running totals ----------------------------
    let mut r = 0;
    while r < n {
        let row = rows[r];
        if state[r] == ACTIVE {
            for (body, linear, angular) in [
                (row.a, row.linear_a, row.angular_a),
                (row.b, row.linear_b, row.angular_b),
            ] {
                if body == NONE {
                    continue;
                }
                let local = &mut locals[body as usize];
                local.linear = add(local.linear, scale(linear, moving[r]));
                local.angular = add(local.angular, scale(angular, moving[r]));
                local.free_linear = add(local.free_linear, scale(linear, free[r]));
                local.free_angular = add(local.free_angular, scale(angular, free[r]));
            }
        }
        match row.kind {
            Kind::Pair { contact } if state[r] == ACTIVE => {
                let total = lambda[r].max(0.0);
                let mut spent = contact_impulse.get(contact as usize);
                spent.normal += total;
                spent.driven += moving[r].clamp(0.0, total);
                contact_impulse.set(contact as usize, spent);
            }
            // Both ends of a patch are adjacent rows, end 0 first.
            Kind::Ground { patch, end: 0 } => {
                let mut patch_load = [0.0f64; 2];
                let mut driven = 0.0;
                for e in 0..2 {
                    if state[r + e] == ACTIVE {
                        let total = lambda[r + e].max(0.0);
                        patch_load[e] = total;
                        driven += moving[r + e].clamp(0.0, total);
                    }
                }
                let total = patch_load[0] + patch_load[1];
                if total > 0.0 {
                    let contact = grounds[patch as usize];
                    let body = &locals[row.a as usize].gathered.now;
                    let arm = |e: usize| rotate(body.orientation, contact.local[e]);
                    // Where the load stands, weighted as the two ends carry: exactly
                    // solve_ground_normal's arm, so friction and the rolling couple act
                    // where the normal force is.
                    let load = scale(
                        add(scale(arm(0), patch_load[0]), scale(arm(1), patch_load[1])),
                        1.0 / total,
                    );
                    let mut totals = ground_impulse.get(patch as usize);
                    totals.spent.normal += total;
                    totals.spent.driven += driven;
                    totals.local_load = rotate_inv(body.orientation, load);
                    ground_impulse.set(patch as usize, totals);
                }
                // The ends' bodies are written below with every other row's; skip on to
                // the second end's own body share.
                if state[r + 1] == ACTIVE {
                    let other = rows[r + 1];
                    let local = &mut locals[other.a as usize];
                    local.linear = add(local.linear, scale(other.linear_a, moving[r + 1]));
                    local.angular = add(local.angular, scale(other.angular_a, moving[r + 1]));
                    local.free_linear = add(local.free_linear, scale(other.linear_a, free[r + 1]));
                    local.free_angular =
                        add(local.free_angular, scale(other.angular_a, free[r + 1]));
                }
                r += 2;
                continue;
            }
            _ => {}
        }
        r += 1;
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

/// The tail of the factor, over the tail rows that are active, from the Schur complement
/// the head left. A row whose pivot falls to [`SATURATED`] of its own diagonal in `K`
/// depends on the rows before it and is marked so for the rest of the pass.
fn factorise_tail(
    matrix: &[f64],
    factor: &mut [f64],
    schur: &[f64],
    state: &mut [u8],
    n: usize,
    head: usize,
) {
    let tail = n - head;
    for r in head..n {
        for c in head..n {
            factor[r * n + c] = 0.0;
        }
    }
    for i in 0..tail {
        let r = head + i;
        if state[r] != ACTIVE {
            continue;
        }
        for j in 0..=i {
            let c = head + j;
            if state[c] != ACTIVE {
                continue;
            }
            let mut sum = schur[i * tail + j];
            for m in head..c {
                sum -= factor[r * n + m] * factor[c * n + m];
            }
            if c == r {
                let diagonal = matrix[r * n + r];
                if diagonal <= 0.0 || sum <= SATURATED * diagonal {
                    state[r] = DEPENDENT;
                    for m in head..r {
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

/// `L L^T x = b` over the active rows; every other row's `x` is zero.
fn substitute(factor: &[f64], state: &[u8], b: &[f64], x: &mut [f64], y: &mut [f64], n: usize) {
    for r in 0..n {
        if state[r] != ACTIVE {
            y[r] = 0.0;
            continue;
        }
        let mut sum = b[r];
        for c in 0..r {
            sum -= factor[r * n + c] * y[c];
        }
        y[r] = sum / factor[r * n + r];
    }
    for r in (0..n).rev() {
        if state[r] != ACTIVE {
            x[r] = 0.0;
            continue;
        }
        let mut sum = y[r];
        for c in r + 1..n {
            sum -= factor[c * n + r] * x[c];
        }
        x[r] = sum / factor[r * n + r];
    }
}

/// Probe counters: rows, dependent rows and solves per rig per pass. Measurement only.
pub mod probe {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static SOLVES: AtomicU64 = AtomicU64::new(0);
    pub static RIG_PASSES: AtomicU64 = AtomicU64::new(0);
    pub static ROWS: AtomicU64 = AtomicU64::new(0);
    pub static ROWS_MAX: AtomicU64 = AtomicU64::new(0);
    pub static DEPENDENT: AtomicU64 = AtomicU64::new(0);
    pub static HIST: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
    pub fn read() -> [u64; 5] {
        [
            RIG_PASSES.load(Relaxed),
            ROWS.load(Relaxed),
            ROWS_MAX.load(Relaxed),
            DEPENDENT.load(Relaxed),
            SOLVES.load(Relaxed),
        ]
    }
    pub fn hist() -> [u64; 16] {
        std::array::from_fn(|i| HIST[i].load(Relaxed))
    }
    pub(super) fn add(n: usize, dependent: usize, solves: usize) {
        RIG_PASSES.fetch_add(1, Relaxed);
        ROWS.fetch_add(n as u64, Relaxed);
        ROWS_MAX.fetch_max(n as u64, Relaxed);
        DEPENDENT.fetch_add(dependent as u64, Relaxed);
        SOLVES.fetch_add(solves as u64, Relaxed);
        HIST[(n / 10).min(15)].fetch_add(1, Relaxed);
    }
}

fn tally(n: usize, _head: usize, solves: usize, state: &[u8]) {
    let dependent = state.iter().filter(|&&s| s == DEPENDENT).count();
    probe::add(n, dependent, solves);
}
