// The air fetch variants and side passes for `particle_pool.wgsl`. The host takes the
// section it needs, cut at its marker line.

//== filtered
// Hardware trilinear filtering of a float texture (rgba16float, or rgba32float on a
// device with FLOAT32_FILTERABLE). The texture's x axis is the grid's z (the grid is z
// fastest), so the cells upload with no transpose and the coordinate is p.zyx.
@group(1) @binding(0) var air_tex: texture_3d<f32>;
@group(1) @binding(1) var air_sampler: sampler;

fn sample_air(p: vec3<f32>) -> vec3<f32> {
    let uvw = p.zyx * frame.air_scale.xyz + frame.air_offset.xyz;
    return textureSampleLevel(air_tex, air_sampler, uvw, 0.0).xyz;
}

//== exact
// Eight loads and seven lerps in f32, the same operations in the same order as
// `VelocityGrid::sample` on the CPU, for a device that cannot filter rgba32float or a
// caller that wants the CPU's fetch.
@group(1) @binding(0) var air_tex: texture_3d<f32>;

fn tap(i: i32, j: i32, k: i32) -> vec3<f32> {
    return textureLoad(air_tex, vec3<i32>(k, j, i), 0).xyz;
}

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn sample_air(p: vec3<f32>) -> vec3<f32> {
    var g = (p - frame.air_first_centre.xyz) * frame.air_first_centre.w;
    g = min(max(g, vec3<f32>(0.0)), frame.air_last.xyz);
    let base = floor(min(g, frame.air_last_base.xyz));
    let f = g - base;
    let b = vec3<i32>(base);
    let x0 = lerp3(
        lerp3(tap(b.x, b.y, b.z), tap(b.x, b.y, b.z + 1), f.z),
        lerp3(tap(b.x, b.y + 1, b.z), tap(b.x, b.y + 1, b.z + 1), f.z),
        f.y,
    );
    let x1 = lerp3(
        lerp3(tap(b.x + 1, b.y, b.z), tap(b.x + 1, b.y, b.z + 1), f.z),
        lerp3(tap(b.x + 1, b.y + 1, b.z), tap(b.x + 1, b.y + 1, b.z + 1), f.z),
        f.y,
    );
    return lerp3(x0, x1, f.x);
}

//== convert
// The f32 cells, as `VelocityGrid` stores them, into an rgba16float texture: one
// thread a cell, with the texture's x the grid's z. The device rounds (to nearest or toward zero; Vulkan leaves it to the device).
@group(0) @binding(0) var<storage, read> cells: array<vec4<f32>>;
@group(0) @binding(1) var field_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(64)
fn convert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = textureDimensions(field_out);
    let n = d.x * d.y * d.z;
    let idx = gid.x;
    if (idx >= n) {
        return;
    }
    let k = idx % d.x;
    let j = (idx / d.x) % d.y;
    let i = idx / (d.x * d.y);
    textureStore(field_out, vec3<u32>(k, j, i), cells[idx]);
}

//== probe
// Debug and test only: the air at given points, through the same `sample_air` the
// integrate uses.
@group(0) @binding(8) var<storage, read> probe_in: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&probe_in)) {
        return;
    }
    probe_out[i] = vec4<f32>(sample_air(probe_in[i].xyz), 0.0);
}
