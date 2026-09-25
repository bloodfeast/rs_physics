// The scene build: the statics' inverses, the xz grid of static lists (a count, a scan and
// a fill), the static tops per cell, and the max pyramid. Run once per scene and per
// terrain rect; never per tick. The constants above this line are generated.

@group(0) @binding(0) var<storage, read_write> scene: array<atomic<u32>>;

fn ld(i: u32) -> u32 {
    return atomicLoad(&scene[i]);
}

fn ldf(i: u32) -> f32 {
    return bitcast<f32>(atomicLoad(&scene[i]));
}

fn stf(i: u32, v: f32) {
    atomicStore(&scene[i], bitcast<u32>(v));
}

// An order-preserving map from f32 to u32, so `atomicMax` takes the higher top. 0 is below
// every float and means "no static here".
fn ord(f: f32) -> u32 {
    let b = bitcast<u32>(f);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn unord(u: u32) -> f32 {
    return bitcast<f32>(select(~u, u & 0x7FFFFFFFu, (u & 0x80000000u) != 0u));
}

// Per static: its inverse transform, and its count and top in every cell it covers.
@compute @workgroup_size(64)
fn statics_bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= ld(H_STATICS)) {
        return;
    }
    let cols = ld(H_COLS);
    let base = ld(H_OFF_STATICS) + 16u * s;
    let a0 = vec3<f32>(ldf(base), ldf(base + 1u), ldf(base + 2u));
    let a1 = vec3<f32>(ldf(base + 4u), ldf(base + 5u), ldf(base + 6u));
    let a2 = vec3<f32>(ldf(base + 8u), ldf(base + 9u), ldf(base + 10u));
    let t = vec3<f32>(ldf(base + 3u), ldf(base + 7u), ldf(base + 11u));
    let c0 = cross(a1, a2);
    let c1 = cross(a2, a0);
    let c2 = cross(a0, a1);
    let det = dot(a0, c0);
    var i0 = vec3<f32>(0.0);
    var i1 = vec3<f32>(0.0);
    var i2 = vec3<f32>(0.0);
    if (abs(det) >= 1e-30) {
        let inv = 1.0 / det;
        i0 = vec3<f32>(c0.x, c1.x, c2.x) * inv;
        i1 = vec3<f32>(c0.y, c1.y, c2.y) * inv;
        i2 = vec3<f32>(c0.z, c1.z, c2.z) * inv;
    }
    // The static's record for the query: its inverse, top, bottom, half-width and
    // material, 16 words, so a test against it is one round of loads.
    let out = ld(H_OFF_INV) + 16u * s;
    stf(out, i0.x);
    stf(out + 1u, i0.y);
    stf(out + 2u, i0.z);
    stf(out + 3u, -dot(i0, t));
    stf(out + 4u, i1.x);
    stf(out + 5u, i1.y);
    stf(out + 6u, i1.z);
    stf(out + 7u, -dot(i1, t));
    stf(out + 8u, i2.x);
    stf(out + 9u, i2.y);
    stf(out + 10u, i2.z);
    stf(out + 11u, -dot(i2, t));
    let half_y = 0.5 * (abs(a1.x) + abs(a1.y) + abs(a1.z));
    let top = t.y + half_y;
    stf(out + 12u, top);
    stf(out + 13u, t.y - half_y);
    stf(out + 14u, ldf(base + 13u));
    atomicStore(&scene[out + 15u], ld(base + 12u));

    let r = ld(H_OFF_RANGES) + 4u * s;
    let col0 = ld(r);
    let row0 = ld(r + 1u);
    let width = ld(r + 2u) - col0 + 1u;
    let area = width * (ld(r + 3u) - row0 + 1u);
    let counts = ld(H_OFF_COUNT);
    let tops = ld(H_OFF_TOP);
    for (var k = 0u; k < MAX_CELLS_PER_STATIC; k = k + 1u) {
        if (k >= area) {
            break;
        }
        let cell = (row0 + k / width) * cols + col0 + k % width;
        atomicAdd(&scene[counts + cell], 1u);
        atomicMax(&scene[tops + cell], ord(top));
    }
}

var<workgroup> chunk_sums: array<u32, 256>;

// The exclusive scan of the counts into the heads, in one workgroup: each lane sums a
// contiguous chunk, lane 0 scans the 256 sums, and each lane writes its chunk's heads.
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) lane: u32) {
    let cells = ld(H_COLS) * ld(H_ROWS);
    let chunk = (cells + 255u) / 256u;
    let start = min(lane * chunk, cells);
    let end = min(start + chunk, cells);
    let counts = ld(H_OFF_COUNT);
    let heads = ld(H_OFF_HEADS);
    var sum = 0u;
    for (var k = 0u; k < MAX_SCAN_CHUNK; k = k + 1u) {
        if (start + k >= end) {
            break;
        }
        sum = sum + ld(counts + start + k);
    }
    chunk_sums[lane] = sum;
    workgroupBarrier();
    if (lane == 0u) {
        var run = 0u;
        for (var k = 0u; k < 256u; k = k + 1u) {
            let v = chunk_sums[k];
            chunk_sums[k] = run;
            run = run + v;
        }
        atomicStore(&scene[heads + cells], run);
    }
    workgroupBarrier();
    var run = chunk_sums[lane];
    for (var k = 0u; k < MAX_SCAN_CHUNK; k = k + 1u) {
        if (start + k >= end) {
            break;
        }
        atomicStore(&scene[heads + start + k], run);
        run = run + ld(counts + start + k);
    }
}

// Per static: its index into every covered cell's list. The counts run back down to zero,
// which is what makes each slot unique.
@compute @workgroup_size(64)
fn statics_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= ld(H_STATICS)) {
        return;
    }
    let cols = ld(H_COLS);
    let r = ld(H_OFF_RANGES) + 4u * s;
    let col0 = ld(r);
    let row0 = ld(r + 1u);
    let width = ld(r + 2u) - col0 + 1u;
    let area = width * (ld(r + 3u) - row0 + 1u);
    let counts = ld(H_OFF_COUNT);
    let heads = ld(H_OFF_HEADS);
    let list = ld(H_OFF_LIST);
    for (var k = 0u; k < MAX_CELLS_PER_STATIC; k = k + 1u) {
        if (k >= area) {
            break;
        }
        let cell = (row0 + k / width) * cols + col0 + k % width;
        let slot = ld(heads + cell) + atomicSub(&scene[counts + cell], 1u) - 1u;
        atomicStore(&scene[list + slot], s);
    }
}

override LEVEL: u32 = 1u;

// One level of the max pyramid over the header's rect: the max of the 2 x 2 below. Level 1
// reads the terrain and the static tops together, so a block holds its tallest thing.
@compute @workgroup_size(8, 8)
fn pyramid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rc = ld(H_RECT);
    let rr = ld(H_RECT + 1u);
    let rw = ld(H_RECT + 2u);
    let rh = ld(H_RECT + 3u);
    let c = (rc >> LEVEL) + gid.x;
    let r = (rr >> LEVEL) + gid.y;
    if (c > ((rc + rw - 1u) >> LEVEL) || r > ((rr + rh - 1u) >> LEVEL)) {
        return;
    }
    let out_cols = ld(H_PYR_DIMS + 2u * (LEVEL - 1u));
    var below_cols = ld(H_COLS);
    var below_rows = ld(H_ROWS);
    if (LEVEL > 1u) {
        below_cols = ld(H_PYR_DIMS + 2u * (LEVEL - 2u));
        below_rows = ld(H_PYR_DIMS + 2u * (LEVEL - 2u) + 1u);
    }
    var m = -3.0e38;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let bc = 2u * c + (k & 1u);
        let br = 2u * r + (k >> 1u);
        if (bc < below_cols && br < below_rows) {
            let at = br * below_cols + bc;
            if (LEVEL == 1u) {
                m = max(m, ldf(ld(H_OFF_HEIGHTS) + at));
                let top = ld(ld(H_OFF_TOP) + at);
                if (top != 0u) {
                    m = max(m, unord(top));
                }
            } else {
                m = max(m, ldf(ld(H_OFF_PYR + LEVEL - 2u) + at));
            }
        }
    }
    stf(ld(H_OFF_PYR + LEVEL - 1u) + r * out_cols + c, m);
}
