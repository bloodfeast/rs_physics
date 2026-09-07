#import bevy_pbr::{
    mesh_view_bindings::globals,
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

struct WaterStreamMaterial {
    base_color: vec4<f32>,
    flow_speed: f32,
    turbulence: f32,
    opacity: f32,
    _padding: f32,
};

@group(2) @binding(0)
var<uniform> material: WaterStreamMaterial;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let time = globals.time;
    let uv = in.uv;

    // === FAST FALLING WATER STREAKS ===

    // Time-based vertical scroll - water falls DOWN (UV.y increases downward in Bevy)
    let fall_speed = material.flow_speed * 5.0;
    let fall_offset = time * fall_speed;

    // Create multiple vertical falling streaks at different speeds
    // Streak layer 1 - wide main streaks
    let streak1_x = uv.x * 6.0;
    let streak1_y = (1.0 - uv.y) * 12.0 + fall_offset;
    let streak1 = sin(streak1_x * 3.14159) * 0.5 + 0.5;
    let streak1_flow = fract(streak1_y * 0.5);

    // Streak layer 2 - thinner faster streaks
    let streak2_x = uv.x * 10.0 + 1.5;
    let streak2_y = (1.0 - uv.y) * 20.0 + fall_offset * 1.3;
    let streak2 = sin(streak2_x * 3.14159) * 0.5 + 0.5;
    let streak2_flow = fract(streak2_y * 0.3);

    // Streak layer 3 - very thin highlight streaks
    let streak3_x = uv.x * 15.0 + 0.7;
    let streak3_y = (1.0 - uv.y) * 30.0 + fall_offset * 1.8;
    let streak3 = sin(streak3_x * 3.14159) * 0.5 + 0.5;
    let streak3_flow = fract(streak3_y * 0.2);

    // Combine streaks - the flow creates visible downward motion
    let combined_streaks = streak1 * 0.5 + streak2 * 0.3 + streak3 * 0.2;

    // Add flowing bright bands that travel down
    let band1 = smoothstep(0.4, 0.5, streak1_flow) * smoothstep(0.6, 0.5, streak1_flow);
    let band2 = smoothstep(0.3, 0.4, streak2_flow) * smoothstep(0.5, 0.4, streak2_flow);
    let band3 = smoothstep(0.35, 0.45, streak3_flow) * smoothstep(0.55, 0.45, streak3_flow);

    let flow_bands = band1 * streak1 * 0.6 + band2 * streak2 * 0.5 + band3 * streak3 * 0.8;

    // === WATER COLOR ===

    // Base water color - cyan/teal
    let deep_water = vec3<f32>(0.1, 0.35, 0.55);
    let light_water = vec3<f32>(0.4, 0.7, 0.9);

    // Color based on streak density
    var color = mix(light_water, deep_water, combined_streaks * 0.5);

    // Add bright flowing highlights (white/light blue bands moving down)
    let highlight_color = vec3<f32>(0.85, 0.95, 1.0);
    color = mix(color, highlight_color, flow_bands * 0.7);

    // Extra bright thin streaks
    let thin_highlights = smoothstep(0.7, 0.9, streak3) * band3;
    color = mix(color, vec3<f32>(1.0, 1.0, 1.0), thin_highlights * 0.5);

    // Foam at top where water pours over
    let top_foam = smoothstep(0.15, 0.0, 1.0 - uv.y);
    color = mix(color, vec3<f32>(0.9, 0.95, 1.0), top_foam * 0.6);

    // === ALPHA ===

    // High base opacity
    var alpha = material.opacity;

    // Horizontal edge fade
    let edge_x = 1.0 - pow(abs(uv.x - 0.5) * 2.0, 1.5);
    alpha *= mix(0.4, 1.0, edge_x);

    // Bottom breaks into droplets
    let bottom_fade = 1.0 - smoothstep(0.8, 1.0, 1.0 - uv.y) * (1.0 - combined_streaks) * 0.5;
    alpha *= bottom_fade;

    // Vary with streaks
    alpha *= 0.75 + combined_streaks * 0.25;

    if (alpha < 0.1) {
        discard;
    }

    return vec4<f32>(color, alpha);
}
