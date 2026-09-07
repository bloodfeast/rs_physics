#import bevy_pbr::{
    mesh_view_bindings::globals,
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

struct SplashMaterial {
    base_color: vec4<f32>,
    highlight_color: vec4<f32>,
    fresnel_power: f32,
    opacity: f32,
    _padding1: f32,
    _padding2: f32,
};

@group(2) @binding(0)
var<uniform> material: SplashMaterial;

const WATER_IOR: f32 = 1.333;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let time = globals.time;
    let world_pos = in.world_position.xyz;
    let normal = normalize(in.world_normal);

    // Camera
    let view_pos = view.world_position;
    let view_dir = normalize(view_pos - world_pos);

    // Fresnel - bright edges like a water droplet
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, material.fresnel_power);

    // Rim lighting for that droplet shine
    let rim = pow(1.0 - n_dot_v, 3.0) * 0.8;

    // Sun direction for specular
    let sun_dir = normalize(vec3<f32>(0.3, 0.8, 0.2));
    let half_vec = normalize(sun_dir + view_dir);
    let spec = pow(max(dot(normal, half_vec), 0.0), 64.0);

    // Internal caustic-like pattern (fake refraction)
    let caustic_uv = normal.xy * 2.0 + time * 0.5;
    let caustic = sin(caustic_uv.x * 10.0) * sin(caustic_uv.y * 10.0) * 0.1 + 0.1;

    // Base water droplet color - slightly darker in center, brighter at edges
    var color = mix(
        material.base_color.rgb * 0.8,
        material.base_color.rgb,
        fresnel * 0.5 + 0.5
    );

    // Add internal light scattering (subsurface-like)
    color += material.base_color.rgb * caustic * n_dot_v;

    // Add rim highlight
    color += material.highlight_color.rgb * rim;

    // Add specular highlight
    color += material.highlight_color.rgb * spec * 0.6;

    // Soft edge fade - droplets should have soft boundaries
    let edge_fade = smoothstep(0.0, 0.3, n_dot_v);

    // Final alpha - fresnel makes edges more visible, but soft fade at grazing angles
    let alpha = material.opacity * mix(0.4, 0.9, fresnel) * edge_fade;

    return vec4<f32>(color, alpha);
}
