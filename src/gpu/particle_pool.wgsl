// The resident effect-particle pool: emission into free slots, and the integrate.
//
// Mirrors `ParticleEffects::integrate_in_air` (src/particles/particle_effects.rs) one
// operation for one operation, except that the air is sampled every frame (no stored
// sample, no stagger) and ground contact and retirement happen in the same pass.
//
// Slots are stable: a particle keeps its slot from emission to retirement. A slot is
// live while `pos_life[i].w > 0`. Every slot below the capacity is either live or on
// the free stack, exactly once.
//
// The host prepends the air-sampling function and its bindings (group 1, bindings 0
// and 1) for the field format it chose, and appends `particle_pool_emit.wgsl`, which
// expands the frame's emission (records and burst descriptors) into particles; see
// `particle_pool.rs`.

const WG: u32 = 64u;
const MAX_CLASSES: u32 = 8u;
// A quarter of an f32 spacing, relative to the power of two below a value: 2^-25.
const ROUNDS_AWAY: f32 = 2.98023224e-8;
const MIN_POSITIVE: f32 = 1.17549435e-38;
const EXPONENT: u32 = 0x7f800000u;
// Tangential friction on ground contact, as `collide_ground_with`.
const GROUND_FRICTION: f32 = 0.55;

// Set in a slot's class word by `place_free` for the frame the particle is placed, so
// `place_overwrite` never replaces a particle placed the same frame; the integrate
// clears it.
const BORN: u32 = 0x80000000u;

const FLAG_AIR: u32 = 1u;
const FLAG_GROUND: u32 = 2u;

// Words per landing: position 3, impact speed, class, size.
const LANDING_WORDS: u32 = 6u;

struct Frame {
    dt: f32,
    emit_count: u32,
    capacity: u32,
    flags: u32,
    landing_capacity: u32,
    sprite_vertices: u32,
    pad0: u32,
    pad1: u32,
    // Filtered fetch: texture coordinate = p.zyx * air_scale + air_offset.
    air_scale: vec4<f32>,
    air_offset: vec4<f32>,
    // Exact fetch: centre of cell (0, 0, 0) in xyz, 1 / h in w.
    air_first_centre: vec4<f32>,
    // dims - 1 and dims - 2 per axis, x y z.
    air_last: vec4<f32>,
    air_last_base: vec4<f32>,
    // Ground: min x, min z, cell size, unused.
    ground: vec4<f32>,
    // Ground: corners - 1 across (x) and down (z).
    ground_last: vec4<i32>,
    // Per class: gravity * dt, damping 1 - min(drag dt, 1), 1 where y only decays,
    // the air's share swirl * max(min(drag dt, 1), 0).
    rows: array<vec4<f32>, 8>,
    // Per class restitution, packed four a vector.
    restitution: array<vec4<f32>, 2>,
    // Emission: segments in the table at word 0 of `records`; the word in `records`
    // where the jump tables start (the sine table follows them); hex digits the jump
    // tables cover; and a zero the compiler cannot see, for `rounded`.
    emit_segments: u32,
    table_base: u32,
    jump_levels: u32,
    zero: u32,
}

struct State {
    free_count: atomic<u32>,
    live: atomic<u32>,
    high_water: atomic<u32>,
    cursor: atomic<u32>,
    landings: atomic<u32>,
    retired: atomic<u32>,
    placed: atomic<u32>,
    overwritten: atomic<u32>,
    dropped: atomic<u32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> records: array<u32>;
@group(0) @binding(2) var<storage, read_write> pos_life: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> class_size: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> free: array<u32>;
@group(0) @binding(6) var<storage, read_write> state: State;
@group(0) @binding(7) var<storage, read_write> landings: array<u32>;

@group(1) @binding(2) var ground_tex: texture_2d<f32>;

// Indirect arguments: [0..3) the integrate's dispatch, [3..6) the overwrite placement's
// dispatch, [8..12) a non-indexed draw over the slots in use.
@group(2) @binding(0) var<storage, read_write> args: array<u32>;

fn emit_zero() -> u32 {
    return frame.zero;
}

// The frame's particle `r` (0 the oldest) into `slot`.
fn write_record(slot: u32, r: u32, born: u32) {
    let jump_words = frame.jump_levels * DIGIT_TABLES * JUMP_TABLE_WORDS;
    let p = emitted(r, frame.emit_segments, frame.table_base, frame.jump_levels, frame.table_base + jump_words);
    pos_life[slot] = vec4<f32>(p.position, p.remaining);
    vel[slot] = vec4<f32>(p.velocity, p.lifetime);
    // Class, then the size's bits.
    class_size[slot] = vec2<u32>(p.class_id | born, bitcast<u32>(p.size));
}

var<workgroup> wg_high: atomic<u32>;

// New particles into free slots: the newest `min(e, f)` records take slots off the top of
// the free stack. The older records past the free count are left to `place_overwrite`.
// Reads the free count; `finalize` lowers it after every placement has read it.
@compute @workgroup_size(WG)
fn place_free(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = gid.x;
    let e = frame.emit_count;
    let f = atomicLoad(&state.free_count);
    if (i == 0u) {
        // How many records find no free slot: those overwrite, in the next dispatch.
        let over = select(0u, e - f, e > f);
        args[3] = (over + WG - 1u) / WG;
        args[4] = 1u;
        args[5] = 1u;
    }
    if (lid == 0u) {
        atomicStore(&wg_high, 0u);
    }
    workgroupBarrier();
    let taken = min(e, f);
    if (i < taken) {
        let slot = free[f - 1u - i];
        write_record(slot, e - taken + i, BORN);
        atomicMax(&wg_high, slot + 1u);
    }
    workgroupBarrier();
    if (lid == 0u) {
        atomicMax(&state.high_water, atomicLoad(&wg_high));
    }
}

// Records that found no free slot (the oldest of the frame's) replace live particles at
// a rotating cursor, the rotation `ParticleEffects` uses when full. A slot `place_free`
// filled this frame is not replaced: the older record is dropped instead, so the newest
// particles are the ones kept. Runs after `place_free`, so every write is whole.
// Dispatched indirectly with the count `place_free` wrote: zero workgroups on any frame
// the pool has room.
@compute @workgroup_size(WG)
fn place_overwrite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let e = frame.emit_count;
    let f = atomicLoad(&state.free_count);
    if (e <= f || j >= e - f) {
        return;
    }
    let slot = (atomicLoad(&state.cursor) + j) % frame.capacity;
    if ((class_size[slot].x & BORN) != 0u) {
        atomicAdd(&state.dropped, 1u);
        return;
    }
    write_record(slot, j, 0u);
    atomicAdd(&state.placed, 1u);
    atomicAdd(&state.overwritten, 1u);
}

// One thread: settles the counts for this frame's placements, resets the landings and
// writes the integrate's dispatch and the draw's arguments from the slots in use.
@compute @workgroup_size(1)
fn finalize() {
    let e = frame.emit_count;
    let f = atomicLoad(&state.free_count);
    let taken = min(e, f);
    let over = e - taken;
    atomicStore(&state.free_count, f - taken);
    atomicAdd(&state.live, taken);
    atomicAdd(&state.placed, taken);
    atomicStore(&state.cursor, (atomicLoad(&state.cursor) + over) % frame.capacity);
    atomicStore(&state.landings, 0u);
    let high = atomicLoad(&state.high_water);
    args[0] = (high + WG - 1u) / WG;
    args[1] = 1u;
    args[2] = 1u;
    args[8] = frame.sprite_vertices;
    args[9] = high;
    args[10] = 0u;
    args[11] = 0u;
}

fn resolution(p: f32) -> f32 {
    return max(bitcast<f32>(bitcast<u32>(p) & EXPONENT) * ROUNDS_AWAY, MIN_POSITIVE);
}

fn corner(ix: i32, iz: i32) -> f32 {
    let x = clamp(ix, 0, frame.ground_last.x);
    let z = clamp(iz, 0, frame.ground_last.y);
    return textureLoad(ground_tex, vec2<i32>(x, z), 0).x;
}

// The ground height at (x, z), bilinear between the four corners round it, as the
// engine's heightfield computes it on the CPU.
fn ground_height(x: f32, z: f32) -> f32 {
    let u = (x - frame.ground.x) / frame.ground.z;
    let v = (z - frame.ground.y) / frame.ground.z;
    let fu = floor(u);
    let fv = floor(v);
    let fx = u - fu;
    let fz = v - fv;
    let ix = i32(fu);
    let iz = i32(fv);
    let top = corner(ix, iz) * (1.0 - fx) + corner(ix + 1, iz) * fx;
    let bottom = corner(ix, iz + 1) * (1.0 - fx) + corner(ix + 1, iz + 1) * fx;
    return top * (1.0 - fz) + bottom * fz;
}

fn restitution_of(c: u32) -> f32 {
    return frame.restitution[c / 4u][c % 4u];
}

@compute @workgroup_size(WG)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    var a = pos_life[i];
    // Dead (or never used): nothing to do. Written as a negation so NaN counts as dead.
    if (!(a.w > 0.0)) {
        return;
    }
    let vw = vel[i];
    let m = class_size[i];
    let c = m.x & (MAX_CLASSES - 1u);
    if ((m.x & BORN) != 0u) {
        class_size[i].x = m.x & ~BORN;
    }
    let row = frame.rows[c];
    let dt = frame.dt;

    var x = vw.x * row.y;
    var y = (vw.y - row.x) * row.y;
    var z = vw.z * row.y;
    // Selected, not added with a zero weight: a class that ignores the air keeps every
    // bit of the plain update.
    if (row.w != 0.0 && (frame.flags & FLAG_AIR) != 0u) {
        let u = sample_air(a.xyz);
        x = x + row.w * u.x;
        y = y + row.w * u.y;
        z = z + row.w * u.z;
    }

    // A component too small to move its particle is zero (see `integrate_free_flight`).
    let sx = x * dt;
    let sy = y * dt;
    let sz = z * dt;
    if (abs(sx) < resolution(a.x)) { x = 0.0; }
    if (abs(sy) < resolution(a.y) * row.z) { y = 0.0; }
    if (abs(sz) < resolution(a.z)) { z = 0.0; }

    var v = vec3<f32>(x, y, z);
    a = vec4<f32>(a.x + x * dt, a.y + y * dt, a.z + z * dt, a.w - dt);

    if (!(a.w > 0.0)) {
        // Retired: the slot goes back on the free stack.
        pos_life[i] = a;
        vel[i] = vec4<f32>(v, vw.w);
        let top = atomicAdd(&state.free_count, 1u);
        free[top] = i;
        atomicSub(&state.live, 1u);
        atomicAdd(&state.retired, 1u);
        return;
    }

    if ((frame.flags & FLAG_GROUND) != 0u) {
        let floor_y = ground_height(a.x, a.z);
        if (a.y < floor_y) {
            // Reported before the bounce, so the speed is the one it arrived at.
            let k = atomicAdd(&state.landings, 1u);
            if (k < frame.landing_capacity) {
                let b = k * LANDING_WORDS;
                landings[b] = bitcast<u32>(a.x);
                landings[b + 1u] = bitcast<u32>(floor_y);
                landings[b + 2u] = bitcast<u32>(a.z);
                landings[b + 3u] = bitcast<u32>(-min(v.y, 0.0));
                landings[b + 4u] = c;
                landings[b + 5u] = m.y;
            }
            a.y = floor_y;
            v.y = -v.y * restitution_of(c);
            v.x = v.x * GROUND_FRICTION;
            v.z = v.z * GROUND_FRICTION;
        }
    }

    pos_life[i] = a;
    vel[i] = vec4<f32>(v, vw.w);
}
