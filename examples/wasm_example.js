import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import fs from 'fs/promises';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let wasm_module;
let wasm_bytes;

if (typeof window === 'undefined') {
    // Node.js environment
    const wasm_path = resolve(__dirname, '../rs_physics_wasm/pkg/rs_physics_wasm_bg.wasm');
    wasm_bytes = await fs.readFile(wasm_path);

    const js_path = resolve(__dirname, '../rs_physics_wasm/pkg/rs_physics_wasm.js');
    const js_url = `file://${js_path.replace(/\\/g, '/')}`;
    wasm_module = await import(js_url);
} else {
    // Browser environment
    wasm_module = await import('../rs_physics_wasm/pkg/rs_physics_wasm.js');
}

const { default: init, WasmPhysics } = wasm_module;

async function runSimulation() {
    if (typeof window === 'undefined') {
        // Node.js environment: pass the WASM bytes directly
        await init(wasm_bytes);
    } else {
        // Browser environment: let it fetch the WASM file
        await init();
    }

    const physics = new WasmPhysics();

    // Create two objects
    const obj1 = physics.create_object(1.0, 5.0, 0.0);
    const obj2 = physics.create_object(2.0, -3.0, 10.0);

    console.log("Initial state:");
    console.log(`Object 1: mass=${obj1.mass}, velocity=${obj1.velocity}, position=${obj1.position}`);
    console.log(`Object 2: mass=${obj2.mass}, velocity=${obj2.velocity}, position=${obj2.position}`);

    // Calculate initial energies and momenta
    const initialEnergy1 = physics.calculate_kinetic_energy(obj1);
    const initialEnergy2 = physics.calculate_kinetic_energy(obj2);
    const initialMomentum1 = physics.calculate_momentum(obj1);
    const initialMomentum2 = physics.calculate_momentum(obj2);

    console.log(`Initial energy: ${initialEnergy1 + initialEnergy2} J`);
    console.log(`Initial momentum: ${initialMomentum1 + initialMomentum2} kg⋅m/s`);

    // Simulate a collision
    const angle = 0.0;
    const duration = 0.1;
    const dragCoefficient = 0.47;
    const crossSectionalArea = 1.0;
    physics.simulate_collision(obj1, obj2, angle, duration, dragCoefficient, crossSectionalArea);

    console.log("\nAfter collision:");
    console.log(`Object 1: mass=${obj1.mass}, velocity=${obj1.velocity}, position=${obj1.position}`);
    console.log(`Object 2: mass=${obj2.mass}, velocity=${obj2.velocity}, position=${obj2.position}`);

    // Calculate final energies and momenta
    const finalEnergy1 = physics.calculate_kinetic_energy(obj1);
    const finalEnergy2 = physics.calculate_kinetic_energy(obj2);
    const finalMomentum1 = physics.calculate_momentum(obj1);
    const finalMomentum2 = physics.calculate_momentum(obj2);

    console.log(`Final energy: ${finalEnergy1 + finalEnergy2} J`);
    console.log(`Final momentum: ${finalMomentum1 + finalMomentum2} kg⋅m/s`);

    // Calculate gravitational force between objects
    const force = physics.calculate_gravity_force(obj1, obj2);
    console.log(`\nGravitational force between objects: ${force} N`);
}

runSimulation().catch(console.error);