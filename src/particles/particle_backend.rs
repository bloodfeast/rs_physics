//! Deciding, per frame, whether particles integrate on the CPU or the GPU.
//!
//! # Why this is a policy and not a constant
//!
//! The obvious implementation is `if count > 500_000 { gpu }`. That number came off
//! one benchmark on one machine, and it is wrong everywhere else: a laptop with an
//! integrated GPU and a workstation with an RTX 3090 have crossovers an order of
//! magnitude apart, and the same binary ships to both. Worse, the constant goes
//! stale the moment either code path is optimized — the CPU integrator here is
//! currently scalar, and vectorizing it would move the real crossover by roughly
//! 10x while the constant sat there confidently saying 500,000.
//!
//! So this measures instead. Both paths are timed as they run, the per-particle
//! cost of each is tracked as a moving average, and the choice each frame is
//! whichever *predicted* cost is lower for the population that actually exists this
//! frame. A machine calibrates itself within a few frames of doing real work, and
//! an optimization to either path is picked up automatically rather than requiring
//! someone to remember to re-tune a threshold.
//!
//! # The cost model
//!
//! ```text
//!   cpu(n) = n * cpu_per_particle
//!   gpu(n) = launch_overhead + n * gpu_per_particle + transfer(n)
//! ```
//!
//! `launch_overhead` is the term that decides everything at small `n`. A kernel
//! launch plus a synchronise costs tens of microseconds no matter how little work
//! it does, which is why a thousand particles will never be worth dispatching.
//!
//! `transfer(n)` is the term people forget, and it is usually the one that kills
//! the idea. If the pool has to be uploaded and read back every frame, PCIe
//! bandwidth puts the per-particle cost *above* an optimized CPU loop and the GPU
//! path loses at every population. The GPU only wins when the pool stays
//! **resident** across frames and the renderer draws from the device buffer
//! directly. [`GpuResidency`] makes that a explicit input to the decision rather
//! than an assumption buried in a constant.
//!
//! # Hysteresis
//!
//! Predicted costs cross and re-cross as the population fluctuates, and switching
//! backends is not free — at minimum it means the pool is in the wrong place. So a
//! switch requires the alternative to be better by a margin, not merely better.
//! That gives a dead band around the crossover without needing two hand-tuned
//! thresholds that can be set inconsistently.

use std::time::Duration;

/// Where the integration step ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
}

/// What the caller wants, as opposed to what the policy would choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendMode {
    /// Measure and choose per frame.
    #[default]
    Auto,
    /// Always CPU. Useful for reproducibility and for isolating a GPU bug.
    ForceCpu,
    /// Always GPU where one exists. Falls back to CPU rather than failing.
    ForceGpu,
}

/// Whether the particle pool lives on the device between frames.
///
/// This single fact changes the GPU cost model by more than an order of magnitude,
/// so it is stated rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuResidency {
    /// The pool stays in device memory and the renderer draws from it. Only the
    /// kernel cost applies.
    #[default]
    Resident,
    /// The pool is uploaded and read back each frame. Adds PCIe transfer per
    /// particle, which is frequently enough to make the GPU path lose outright.
    RoundTrip,
}

/// Nanoseconds per particle for the CPU path, before any measurement.
///
/// Seeded from a measured run so the first few frames are not wild; it is replaced
/// by real numbers as soon as the pool does meaningful work.
///
/// 3.0 is `ParticleEffects::integrate` on a live, re-emitting pool (3.19 ns at 16k,
/// 2.88 ns at 100k: `examples/r2_bench.rs`, 2026-09-28, i9-10980XE). It was 15.0 until
/// then, taken from a bench whose never-retiring pool had drifted into subnormal
/// velocities; that seed put the resident crossover near 4,000 particles, about six
/// times too early.
const CPU_SEED_NS: f32 = 3.0;

/// Fixed cost of a frame on the GPU: the launches, and for a round trip a synchronise.
///
/// 10,000 is the resident pool's (`gpu::GpuParticlePool`): its emit and integrate passes
/// on a near-empty pool, 9.7 us of device time a frame by timestamp queries
/// (`examples/pool_gpu_bench.rs`, 2026-10-03, RTX 3090, beside another build). It was
/// 60,000, a launch plus a synchronise, before a resident backend existed; a resident
/// pool never synchronises on the frame path.
const GPU_SEED_FIXED_NS: f32 = 10_000.0;

/// Per-particle kernel cost once the data is already on the device.
///
/// 0.1 is the resident pool's integrate on the live sparks-and-dust pool with the air
/// at 20 Hz: 0.104 ns at 100k and 0.092 ns at 1M, the 1M figure at the card's memory
/// bandwidth (72 bytes a particle). It was 0.4, an estimate.
const GPU_SEED_NS: f32 = 0.1;

/// Per-particle cost of moving a particle across PCIe and back. Ten `f32` fields
/// each way over a bus an order of magnitude slower than device memory.
const GPU_SEED_TRANSFER_NS: f32 = 6.5;

/// How much better the alternative must be before switching. 1.25 means "at least
/// 25% cheaper", which is wide enough to sit still through ordinary frame-to-frame
/// jitter and narrow enough to react to a real change in load.
const SWITCH_MARGIN: f32 = 1.25;

/// Weight of a new sample in the moving average. Low enough to ignore a single
/// noisy frame, high enough to track a genuine change in a handful of frames.
const EWMA_ALPHA: f32 = 0.15;

/// A measurement below this is mostly clock resolution, so it is recorded but not
/// trusted to update the model.
const MIN_TRUSTWORTHY_NS: f32 = 2_000.0;

/// Chooses a backend per frame from measured per-particle costs.
///
/// **With the resident GPU pool, the choice is made once, at startup.** A host that has
/// `gpu::GpuParticlePool` (features `gpu` and `particles`) and an adapter runs its effect
/// particles there for the whole run, and on a CPU `ParticleEffects` only when no adapter
/// exists. It does not switch per frame: moving particles from the device back to the CPU
/// needs a readback, which costs a frame, and no such path exists. The per-frame choice
/// below serves hosts without a resident pool, and the costs it holds (the GPU's are
/// seeded from the resident pool's measured figures) are what a HUD reports.
#[derive(Debug, Clone)]
pub struct BackendPolicy {
    mode: BackendMode,
    residency: GpuResidency,
    gpu_available: bool,

    cpu_per_particle_ns: f32,
    gpu_per_particle_ns: f32,
    gpu_fixed_ns: f32,
    gpu_transfer_ns: f32,

    current: Backend,
    cpu_samples: u32,
    gpu_samples: u32,
    switches: u32,
}

impl Default for BackendPolicy {
    fn default() -> Self {
        BackendPolicy {
            mode: BackendMode::Auto,
            residency: GpuResidency::Resident,
            gpu_available: false,
            cpu_per_particle_ns: CPU_SEED_NS,
            gpu_per_particle_ns: GPU_SEED_NS,
            gpu_fixed_ns: GPU_SEED_FIXED_NS,
            gpu_transfer_ns: GPU_SEED_TRANSFER_NS,
            current: Backend::Cpu,
            cpu_samples: 0,
            gpu_samples: 0,
            switches: 0,
        }
    }
}

impl BackendPolicy {
    pub fn new(mode: BackendMode) -> BackendPolicy {
        BackendPolicy {
            mode,
            ..Default::default()
        }
    }

    pub fn set_mode(&mut self, mode: BackendMode) {
        self.mode = mode;
    }

    pub fn mode(&self) -> BackendMode {
        self.mode
    }

    pub fn set_residency(&mut self, residency: GpuResidency) {
        self.residency = residency;
    }

    /// Declare whether a GPU backend actually exists. Until this is true the policy
    /// will never choose one, whatever the numbers say or the mode asks for.
    pub fn set_gpu_available(&mut self, available: bool) {
        self.gpu_available = available;
        if !available {
            self.current = Backend::Cpu;
        }
    }

    pub fn gpu_available(&self) -> bool {
        self.gpu_available
    }

    pub fn current(&self) -> Backend {
        self.current
    }

    /// How many times the backend has changed. A number that climbs every frame
    /// means the margin is too tight for this workload.
    pub fn switch_count(&self) -> u32 {
        self.switches
    }

    /// Predicted cost of each path at a given population, in nanoseconds.
    pub fn predict(&self, count: usize) -> (f32, f32) {
        let n = count as f32;
        let cpu = n * self.cpu_per_particle_ns;

        let transfer = match self.residency {
            GpuResidency::Resident => 0.0,
            GpuResidency::RoundTrip => n * self.gpu_transfer_ns,
        };
        let gpu = self.gpu_fixed_ns + n * self.gpu_per_particle_ns + transfer;

        (cpu, gpu)
    }

    /// The population at which the two paths currently cost the same.
    ///
    /// Exposed because it is the single most useful number for a developer deciding
    /// whether a GPU backend is worth building at all — and for a HUD, where
    /// watching it move as the code changes is worth more than any static figure in
    /// a document.
    ///
    /// `None` when the GPU can never win: with a round-trip pool and a fast CPU
    /// loop, the per-particle terms alone settle it and no population is large
    /// enough to make up the difference.
    pub fn crossover(&self) -> Option<usize> {
        let gpu_marginal = self.gpu_per_particle_ns
            + match self.residency {
                GpuResidency::Resident => 0.0,
                GpuResidency::RoundTrip => self.gpu_transfer_ns,
            };

        let advantage = self.cpu_per_particle_ns - gpu_marginal;
        if advantage <= 0.0 {
            return None;
        }
        Some((self.gpu_fixed_ns / advantage).ceil() as usize)
    }

    /// Pick a backend for this frame.
    pub fn choose(&mut self, count: usize) -> Backend {
        let chosen = match self.mode {
            BackendMode::ForceCpu => Backend::Cpu,
            BackendMode::ForceGpu if self.gpu_available => Backend::Gpu,
            BackendMode::ForceGpu => Backend::Cpu,
            BackendMode::Auto => self.choose_by_cost(count),
        };

        if chosen != self.current {
            self.switches += 1;
            self.current = chosen;
        }
        chosen
    }

    fn choose_by_cost(&self, count: usize) -> Backend {
        if !self.gpu_available {
            return Backend::Cpu;
        }

        let (cpu, gpu) = self.predict(count);

        // Staying put is the default; moving requires clearing the margin. This is
        // the whole of the hysteresis, and it is why the population can wander
        // across the crossover without the backend flapping with it.
        match self.current {
            Backend::Cpu if gpu * SWITCH_MARGIN < cpu => Backend::Gpu,
            Backend::Gpu if cpu * SWITCH_MARGIN < gpu => Backend::Cpu,
            unchanged => unchanged,
        }
    }

    /// Feed a real measurement back into the model.
    ///
    /// Called with whatever the step actually cost. Populations too small to time
    /// reliably are ignored rather than allowed to poison the average with clock
    /// resolution.
    pub fn record(&mut self, backend: Backend, count: usize, elapsed: Duration) {
        if count == 0 {
            return;
        }
        let total_ns = elapsed.as_nanos() as f32;
        if total_ns < MIN_TRUSTWORTHY_NS {
            return;
        }

        match backend {
            Backend::Cpu => {
                let sample = total_ns / count as f32;
                self.cpu_per_particle_ns =
                    blend(self.cpu_per_particle_ns, sample, self.cpu_samples);
                self.cpu_samples = self.cpu_samples.saturating_add(1);
            }
            Backend::Gpu => {
                // Attribute the fixed cost first; what remains is per-particle. A
                // single timing cannot separate the two, so the launch overhead is
                // held at its calibrated value and the marginal cost absorbs the
                // rest.
                let marginal = (total_ns - self.gpu_fixed_ns).max(0.0) / count as f32;
                self.gpu_per_particle_ns =
                    blend(self.gpu_per_particle_ns, marginal, self.gpu_samples);
                self.gpu_samples = self.gpu_samples.saturating_add(1);
            }
        }
    }

    /// Record the measured cost of an empty or near-empty dispatch, which isolates
    /// launch overhead from per-particle work.
    pub fn record_launch_overhead(&mut self, elapsed: Duration) {
        let ns = elapsed.as_nanos() as f32;
        self.gpu_fixed_ns = blend(self.gpu_fixed_ns, ns, self.gpu_samples);
    }

    pub fn cpu_per_particle_ns(&self) -> f32 {
        self.cpu_per_particle_ns
    }

    pub fn gpu_per_particle_ns(&self) -> f32 {
        self.gpu_per_particle_ns
    }
}

/// Exponential moving average that converges quickly from a seeded starting value.
///
/// The first few real samples are weighted heavily, because the seed is a guess and
/// deserves to be overwritten; later ones settle into a steady smoothing factor so
/// one slow frame does not move the decision.
fn blend(current: f32, sample: f32, samples_so_far: u32) -> f32 {
    let alpha = if samples_so_far < 4 {
        0.5
    } else {
        EWMA_ALPHA
    };
    current * (1.0 - alpha) + sample * alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> BackendPolicy {
        let mut p = BackendPolicy::new(BackendMode::Auto);
        p.set_gpu_available(true);
        p
    }

    /// Asserted against the model's own crossover rather than a hardcoded
    /// population, because the crossover *moves* — that is the entire point of the
    /// design. An earlier version of this test hardcoded 10,000 and failed, and it
    /// was the test that was wrong: the seeded costs decide where the crossover sits
    /// (about 3,400 for a resident pool since the GPU seeds were measured on the resident
    /// pool; 23,000 before that, and near 4,000 before the CPU seed was re-measured), and
    /// the test must follow them.
    #[test]
    fn populations_either_side_of_the_crossover_go_the_right_way() {
        let mut p = ready();
        let crossover = p.crossover().expect("resident gpu should win eventually");

        assert_eq!(p.choose(crossover / 8), Backend::Cpu);
        assert_eq!(p.choose(crossover * 8), Backend::Gpu);
    }

    /// A frame's launches cost about ten microseconds no matter how little work they do, so
    /// there is always *some* population too small to be worth dispatching.
    #[test]
    fn a_handful_of_particles_is_never_worth_a_launch() {
        let mut p = ready();
        assert_eq!(p.choose(1), Backend::Cpu);
        assert_eq!(p.choose(100), Backend::Cpu);
    }

    #[test]
    fn large_populations_move_to_the_gpu() {
        let mut p = ready();
        assert_eq!(p.choose(5_000_000), Backend::Gpu);
    }

    #[test]
    fn no_gpu_means_never_gpu_whatever_the_mode() {
        let mut p = BackendPolicy::new(BackendMode::ForceGpu);
        assert!(!p.gpu_available());
        assert_eq!(p.choose(50_000_000), Backend::Cpu);

        p.set_mode(BackendMode::Auto);
        assert_eq!(p.choose(50_000_000), Backend::Cpu);
    }

    #[test]
    fn forcing_a_backend_overrides_the_measurements() {
        let mut p = BackendPolicy::new(BackendMode::ForceCpu);
        p.set_gpu_available(true);
        assert_eq!(p.choose(10_000_000), Backend::Cpu);

        p.set_mode(BackendMode::ForceGpu);
        assert_eq!(p.choose(1), Backend::Gpu);
    }

    /// The failure this design exists to prevent: a population oscillating around
    /// the crossover must not drag the backend with it every frame.
    #[test]
    fn hysteresis_stops_the_backend_flapping() {
        let mut p = ready();
        let crossover = p.crossover().expect("resident gpu should win eventually");

        // Settle onto whatever side we start on, then wobble by a few percent.
        p.choose(crossover);
        let baseline = p.switch_count();

        for i in 0..200 {
            let jitter = if i % 2 == 0 { 97 } else { 103 };
            p.choose(crossover * jitter / 100);
        }

        assert_eq!(
            p.switch_count(),
            baseline,
            "backend flapped across the crossover"
        );
    }

    /// A decisive change in load *should* move it, or the hysteresis is just a
    /// stuck switch.
    #[test]
    fn a_real_change_in_load_still_switches() {
        let mut p = ready();
        assert_eq!(p.choose(1_000), Backend::Cpu);
        assert_eq!(p.choose(20_000_000), Backend::Gpu);
        assert_eq!(p.choose(100), Backend::Cpu);
    }

    /// The point of measuring rather than hard-coding: make the CPU path faster and
    /// the crossover must move out to meet it, with nobody editing a constant.
    #[test]
    fn optimizing_the_cpu_path_moves_the_crossover_out() {
        let mut p = ready();
        let before = p.crossover().unwrap();

        // Report the CPU doing the same work five times faster than its seed. Relative
        // to the seed, so re-measuring the seed does not silently weaken this test.
        let faster = Duration::from_nanos((1_000_000.0 * CPU_SEED_NS / 5.0) as u64);
        for _ in 0..40 {
            p.record(Backend::Cpu, 1_000_000, faster);
        }
        let after = p.crossover().unwrap();

        assert!(
            after > before * 5,
            "crossover barely moved: {before} -> {after}"
        );
    }

    /// Round-tripping the pool over PCIe every frame is usually fatal to the GPU
    /// case, and the model has to say so rather than quietly recommending a
    /// pessimization.
    #[test]
    fn a_round_trip_pool_can_make_the_gpu_never_worth_it() {
        let mut p = ready();
        p.set_residency(GpuResidency::RoundTrip);

        // A CPU loop faster than PCIe per-particle bandwidth.
        for _ in 0..40 {
            p.record(Backend::Cpu, 1_000_000, Duration::from_micros(1_500));
        }

        assert_eq!(
            p.crossover(),
            None,
            "transfer cost should rule the gpu out entirely here"
        );
        assert_eq!(p.choose(10_000_000), Backend::Cpu);
    }

    #[test]
    fn unreliably_short_measurements_do_not_poison_the_model() {
        let mut p = ready();
        let before = p.cpu_per_particle_ns();
        // 100 ns across a million particles is clock noise, not a measurement.
        p.record(Backend::Cpu, 1_000_000, Duration::from_nanos(100));
        assert_eq!(p.cpu_per_particle_ns(), before);
    }

    #[test]
    fn predictions_are_ordered_the_way_the_model_claims() {
        let p = ready();
        let (cpu_small, gpu_small) = p.predict(100);
        assert!(cpu_small < gpu_small, "launch overhead should dominate at n=100");

        let (cpu_big, gpu_big) = p.predict(50_000_000);
        assert!(gpu_big < cpu_big, "bandwidth should dominate at n=50M");
    }
}
