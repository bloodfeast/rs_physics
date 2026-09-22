use super::*;

fn upright(at: (f64, f64, f64), radius: f64, half_length: f64, facets: u32) -> Shape {
    Shape::of(at, Quaternion::identity(), radius, half_length, facets)
}

fn lying(at: (f64, f64, f64), radius: f64, half_length: f64, facets: u32, turn: f64) -> Shape {
    // Laid along x, then rolled about its own length by `turn`.
    let down = Quaternion::from_axis_angle((0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
    let roll = Quaternion::from_axis_angle((1.0, 0.0, 0.0), turn);
    Shape::of(at, roll.multiply(&down).normalized(), radius, half_length, facets)
}

/// **Two prisms clear of each other touch nowhere.** The first thing a separating-axis
/// test has to get right is the separating axis.
#[test]
fn prisms_that_do_not_touch_report_nothing() {
    let a = upright((0.0, 0.0, 0.0), 0.3, 0.5, 8);
    for apart in [0.61f64, 0.8, 1.5, 40.0] {
        let b = upright((apart, 0.0, 0.0), 0.3, 0.5, 8);
        assert!(
            touch(&a, &b).is_none(),
            "two prisms of circumradius 0.3 with their centres {apart} apart were found \
             touching; the widest they reach across is 0.3 each",
        );
    }
    // And apart along the axis rather than across it.
    for apart in [1.01f64, 1.4, 9.0] {
        let b = upright((0.0, apart, 0.0), 0.3, 0.5, 8);
        assert!(
            touch(&a, &b).is_none(),
            "two prisms of half-length 0.5 with their centres {apart} apart along their \
             own axis were found touching",
        );
    }
}

/// **Face against face, and the depth is the overlap.**
///
/// Two octagons sitting flat against each other along a side face: the distance between
/// their centres is twice the inradius less the overlap, and the inradius of a regular
/// octagon of circumradius `R` is `R cos(pi/8)`.
#[test]
fn face_to_face_is_as_deep_as_the_overlap() {
    const R: f64 = 0.3;
    let inradius = R * (std::f64::consts::PI / 8.0).cos();
    let a = upright((0.0, 0.0, 0.0), R, 0.5, 8);
    for overlap in [0.001f64, 0.01, 0.05] {
        // A side face's outward normal for `k = 0` bisects corners 0 and 1, so putting the
        // second prism along that bisector is a face meeting a face.
        let bisector = normalized(add(a.corners[0], a.corners[1])).expect("a face normal");
        let b = Shape::of(
            scale(bisector, 2.0 * inradius - overlap),
            Quaternion::identity(),
            R,
            0.5,
            8,
        );
        let hit = touch(&a, &b).expect("a face resting on a face is a contact");
        let deepest = hit.points[0].expect("a contact has a point").1;
        assert!(
            (deepest - overlap).abs() < 1e-9,
            "two faces overlapping by {overlap:.4} reported {deepest:.6}",
        );
        assert!(
            hit.points[1].is_some(),
            "a face against a face gave one point; the pair has no moment arm and this \
             module exists for the arm",
        );
        // The normal points from `a` to `b`, along the face it is resting on.
        let agrees = dot(hit.normal, bisector);
        assert!(
            agrees > 0.999,
            "the normal is {agrees:.4} of the way along the face it is resting on",
        );
    }
}

/// **Two points, and far enough apart to be a lever.**
///
/// This is the whole claim of the shape: a capsule resting on a capsule gets one point and
/// is free to rotate about it, where a prism resting on a flat gets two and is not. A pair
/// of points a millimetre apart would satisfy the letter of that and none of the physics,
/// so the arm is measured against the body's own size.
#[test]
fn a_face_contact_has_a_real_arm() {
    const R: f64 = 0.3;
    let inradius = R * (std::f64::consts::PI / 8.0).cos();
    let a = lying((0.0, 0.0, 0.0), R, 0.5, 8, 0.0);
    // The second laid the same way, resting on the first's upper face.
    let b = lying((0.0, 2.0 * inradius - 0.01, 0.0), R, 0.5, 8, 0.0);
    let hit = touch(&a, &b).expect("one prism lying on another is a contact");
    let (first, _) = hit.points[0].expect("a contact has a point");
    let (second, _) = hit.points[1].expect("a face contact has two");
    let arm = length(sub(first, second));
    assert!(
        arm > 0.5 * R,
        "the two contact points are {arm:.4} apart on a body of circumradius {R}; that is \
         not a lever, it is one point written twice",
    );
}

/// **A prism is not its bounding capsule**, which is the point of having one.
///
/// Along a face normal a prism reaches its inradius, which is less than its circumradius:
/// two octagons placed so their bounding capsules overlap, but their flats do not, are
/// clear. If this fails the shape is a capsule wearing a different name.
#[test]
fn a_prism_reaches_less_than_its_capsule_across_a_flat() {
    const R: f64 = 0.3;
    let inradius = R * (std::f64::consts::PI / 8.0).cos();
    let a = upright((0.0, 0.0, 0.0), R, 0.5, 8);
    let bisector = normalized(add(a.corners[0], a.corners[1])).expect("a face normal");
    // Between the two: closer than two circumradii, further than two inradii.
    let gap = inradius + 0.5 * (R - inradius);
    assert!(gap < R, "the fixture is not between the two reaches");
    let b = Shape::of(scale(bisector, 2.0 * gap), Quaternion::identity(), R, 0.5, 8);
    assert!(
        touch(&a, &b).is_none(),
        "two prisms whose bounding capsules overlap, but whose flats are {:.4} apart, \
         were found touching",
        2.0 * gap - 2.0 * inradius,
    );
}

/// **Corner into face**, which is the crossed case a pile is mostly made of, and the one
/// the module header's approximation is about. A corner inside a face is a real contact
/// with a real depth, and it is the one case where a single point is the honest answer.
#[test]
fn a_corner_pressed_into_a_face_is_one_deep_point() {
    const R: f64 = 0.3;
    let inradius = R * (std::f64::consts::PI / 8.0).cos();
    // One lying along x, one along z above it, so they cross at a right angle.
    let a = lying((0.0, 0.0, 0.0), R, 0.5, 8, 0.0);
    let across = Quaternion::from_axis_angle((1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2);
    let b = Shape::of(
        (0.0, 2.0 * inradius - 0.02, 0.0),
        across,
        R,
        0.5,
        8,
    );
    let hit = touch(&a, &b).expect("two crossed prisms pressed together is a contact");
    let deepest = hit.points[0].expect("a contact has a point").1;
    assert!(
        deepest > 0.0 && deepest < 0.2,
        "two crossed prisms overlapping by 0.02 reported a depth of {deepest:.4}",
    );
    // Pointing up from the lower one to the upper, which is the only way out.
    assert!(
        hit.normal.1.abs() > 0.9,
        "the normal is {:?}, which is not the way out of a body lying underneath another",
        hit.normal,
    );
}

/// **The answer does not depend on which prism is asked first.** A separating-axis test
/// builds its axes from both bodies, and a contact that is deeper one way round than the
/// other would make the solve depend on the order the broad phase happened to emit a pair.
#[test]
fn the_pair_reads_the_same_either_way_round() {
    const R: f64 = 0.3;
    let a = lying((0.0, 0.0, 0.0), R, 0.5, 8, 0.4);
    let b = lying((0.1, 0.44, 0.05), R, 0.5, 8, 1.1);
    let there = touch(&a, &b).expect("the fixture must be a contact");
    let back = touch(&b, &a).expect("and the same contact the other way round");
    let depth = |hit: &Touch| hit.points[0].map(|(_, d)| d).unwrap_or(0.0);
    assert!(
        (depth(&there) - depth(&back)).abs() < 1e-12,
        "asked one way the pair is {:.9} deep and the other way {:.9}",
        depth(&there),
        depth(&back),
    );
    let opposed = dot(there.normal, back.normal);
    assert!(
        opposed < -0.999999,
        "the two normals are {opposed:.6} of opposite; the way out of a pair cannot \
         depend on which of them was named first",
    );
}

/// **Crossed prisms get two points, which is the case a pile is made of.**
///
/// Ninety per cent of a heap's touching pairs are more than nine degrees from parallel and
/// the median is near fifty, so this is the arrangement that decides whether the shape was
/// worth having. Two crossed prisms meet along one's edge lying across the other's face,
/// which is a *segment* rather than a point: the edge enters the face and leaves it. Two
/// crossed capsules, by contrast, genuinely touch at one point and no amount of solver
/// work gives that point an arm.
///
/// Swept over the angle between them and over the roll of each, because a manifold that
/// happens to have two points at one relative pose and one at the next is an intermittent
/// constraint, and this module's header records four separate occasions on which an
/// intermittent constraint is what a rig walked on.
#[test]
fn crossed_prisms_get_an_arm_at_every_angle() {
    const R: f64 = 0.3;
    let inradius = R * (std::f64::consts::PI / 8.0).cos();
    let mut single = Vec::new();
    for tenth in 2..=16 {
        let crossing = tenth as f64 * 0.1;
        for roll in [0.0f64, 0.2, 0.39] {
            let a = lying((0.0, 0.0, 0.0), R, 0.5, 8, roll);
            let over = Quaternion::from_axis_angle((0.0, 1.0, 0.0), crossing);
            let down = Quaternion::from_axis_angle((0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
            let b = Shape::of(
                (0.0, 2.0 * inradius - 0.01, 0.0),
                over.multiply(&down).normalized(),
                R,
                0.5,
                8,
            );
            let hit = touch(&a, &b).expect("two crossed prisms pressed together touch");
            assert!(hit.points[0].is_some(), "no contact point at all");
            if hit.points[1].is_none() {
                single.push((crossing, roll));
            }
        }
    }
    assert!(
        single.is_empty(),
        "crossed prisms gave a single point, and so no moment arm, at {} of 45 poses: \
         {single:?}",
        single.len(),
    );
}

/// **The cross-section table is the expression it replaced, to the bit.**
///
/// [`cross_section`] exists to keep a `sin` and a `cos` per corner out of a function that
/// runs per candidate pair, and the whole reason it is safe to do that without re-measuring
/// every prism guard is that it changes no answer at all -- not "by less than an epsilon",
/// but not at all. That claim is worth a test of its own, because the cheap mistakes here
/// (folding the angle into `[0, TAU)`, using `k / n` as a fraction of a turn computed a
/// different way, reaching for `sin_cos`) all produce something that is *nearly* this and
/// would pass any tolerance-based comparison while quietly moving every contact normal.
#[test]
fn the_cross_section_table_is_the_expression_it_replaced() {
    for n in 3..=MOST_FACETS as usize {
        let table = cross_section(n);
        assert_eq!(table.len(), n, "the table's row for {n} facets is the wrong length");
        for (k, &(across, along)) in table.iter().enumerate() {
            let turn = std::f64::consts::TAU * k as f64 / n as f64;
            assert_eq!(
                (across, along),
                (turn.cos(), turn.sin()),
                "corner {k} of {n} came out of the table as {across:?}, {along:?} rather \
                 than as the expression it stands in for",
            );
        }
    }
}
