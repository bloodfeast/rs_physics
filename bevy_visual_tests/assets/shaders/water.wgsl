#import bevy_pbr::{
    mesh_view_bindings::globals,
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

struct WaterMaterial {
    deep_color: vec4<f32>,
    shallow_color: vec4<f32>,
    foam_color: vec4<f32>,
    specular_color: vec4<f32>,
    fresnel_power: f32,
    specular_power: f32,
    wave_speed: f32,
    wave_scale: f32,
};

@group(2) @binding(0)
var<uniform> material: WaterMaterial;

const WATER_IOR: f32 = 1.333;
const AIR_IOR: f32 = 1.0;
const POOL_SURFACE_Y: f32 = 2.0;

fn fresnel_schlick(cos_theta: f32) -> f32 {
    let r0 = pow((AIR_IOR - WATER_IOR) / (AIR_IOR + WATER_IOR), 2.0);
    return r0 + (1.0 - r0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let time = globals.time;
    let world_pos = in.world_position.xyz;

    // Get the mesh normal - this contains the wave slope from physics!
    let mesh_normal = normalize(in.world_normal);

    // AMPLIFY the normal perturbation to make waves more visible
    // The mesh normal has subtle x,z components from wave slopes
    // We exaggerate these to make lighting more dramatic
    let normal_strength = 3.0; // Amplification factor
    let amplified_normal = normalize(vec3<f32>(
        mesh_normal.x * normal_strength,
        mesh_normal.y,
        mesh_normal.z * normal_strength
    ));

    // Camera
    let view_pos = view.world_position;
    let view_dir = normalize(view_pos - world_pos);

    // Wave height from actual vertex displacement
    let wave_height = world_pos.y - POOL_SURFACE_Y;

    // Fresnel with amplified normal
    let n_dot_v = max(dot(amplified_normal, view_dir), 0.0);
    let fresnel = fresnel_schlick(n_dot_v);

    // Multiple light sources for more dynamic reflections
    let sun_dir = normalize(vec3<f32>(0.3, 0.8, 0.2));
    let fill_dir = normalize(vec3<f32>(-0.5, 0.6, -0.3));

    // Reflection
    let reflect_dir = reflect(-view_dir, amplified_normal);

    // Sky color based on reflection
    let sky_zenith = vec3<f32>(0.35, 0.55, 0.9);
    let sky_horizon = vec3<f32>(0.75, 0.85, 0.95);
    let sky_color = mix(sky_horizon, sky_zenith, clamp(reflect_dir.y * 0.5 + 0.5, 0.0, 1.0));

    // Water color - varies with wave slope (steeper = darker/deeper looking)
    let slope_factor = 1.0 - abs(mesh_normal.y); // How much the normal deviates from up
    let depth_factor = (1.0 - n_dot_v) * 0.5 + slope_factor * 0.5;
    let water_color = mix(
        material.shallow_color.rgb,
        material.deep_color.rgb,
        depth_factor
    );

    // Specular highlights - toned down for pool water
    let half_sun = normalize(sun_dir + view_dir);
    let half_fill = normalize(fill_dir + view_dir);

    let spec_sun = pow(max(dot(amplified_normal, half_sun), 0.0), material.specular_power * 1.5) * 0.4;
    let spec_fill = pow(max(dot(amplified_normal, half_fill), 0.0), material.specular_power) * 0.1;

    // Sun glitter - subtle sparkles on wave peaks
    let glitter_half = normalize(sun_dir + view_dir);
    let glitter_dot = dot(mesh_normal, glitter_half);
    let glitter = smoothstep(0.995, 1.0, glitter_dot) * 1.0;

    // Secondary glitter - very subtle
    let glitter2_dir = normalize(vec3<f32>(-0.2, 0.9, 0.4));
    let glitter2_half = normalize(glitter2_dir + view_dir);
    let glitter2_dot = dot(mesh_normal, glitter2_half);
    let glitter2 = smoothstep(0.998, 1.0, glitter2_dot) * 0.5;

    // Wave-edge highlighting - subtle rim where waves curve
    let rim_factor = pow(1.0 - n_dot_v, 3.0);
    let rim_light = rim_factor * slope_factor * 0.15;

    // Foam on wave crests - only on tall waves
    let foam = smoothstep(0.15, 0.3, wave_height) * 0.3;

    // Caustics - subtle, animated
    let caustic_scale = 4.0;
    let c1 = noise(world_pos.xz * caustic_scale + time * 0.3);
    let c2 = noise(world_pos.xz * caustic_scale * 1.3 - time * 0.2);
    let caustic = (c1 * c2) * n_dot_v * 0.3;

    // Combine everything
    // Base: blend water color and sky reflection
    var final_color = mix(water_color, sky_color, fresnel * 0.6);

    // Add caustics
    final_color += vec3<f32>(caustic * 0.4, caustic * 0.6, caustic * 0.8);

    // Add specular highlights
    final_color += material.specular_color.rgb * (spec_sun + spec_fill);

    // Add glitter sparkles
    final_color += material.specular_color.rgb * (glitter + glitter2);

    // Add rim lighting on wave edges - blue-tinted, not white
    final_color += vec3<f32>(0.3, 0.5, 0.7) * rim_light;

    // Add foam
    final_color = mix(final_color, material.foam_color.rgb, foam);

    // Trough darkening - where waves dip down, darken slightly
    let trough_darken = smoothstep(0.0, -0.15, wave_height) * 0.2;
    final_color *= (1.0 - trough_darken);

    // Alpha
    let alpha = mix(0.75, 0.92, fresnel);

    return vec4<f32>(final_color, alpha);
}
