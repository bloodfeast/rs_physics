# Acoustics review: `rs_physics::acoustics`

2026-09-29 · base `993f35b`

## Scope

**The ray tracer itself was not reviewed.** The ray-traced acoustics system lives in Ridgeline (`ridgeline/src/audio.rs` and whatever it calls), and that repository was not available to this session. This review covers everything a ray tracer asks `rs_physics` for:

- the medium: speed of sound, ISO 9613-1 absorption, Doppler
- what a surface does to sound: impedance, reflection, absorption
- barriers: Maekawa diffraction
- foliage: ISO 9613-2 Annex A
- where a listener hears a source: Woodworth ITD, head-shadow ILD

The last section lists what to check first in the ray tracer when it can be reviewed.

Every figure was checked against a published table or an independent formula, never against the module's own output. The oracles:

- ISO 9613-2 Table 2 (48 values)
- an independent implementation of ISO 9613-1
- Cramer (1993) for humidity
- Kinsler's tabulated bulk impedances
- ISO 9613-2 §7.3.1 ground classes
- the limits of the Kurze–Anderson form of Maekawa's curve

Tests are in `src/acoustics/regression_tests.rs`, prefixed `review_`. The two `#[ignore]`d ones pin the open limitations: `cargo test --lib acoustics -- --ignored`.

## Summary

The core physics is right. Absorption, the speed of sound, spreading, delay, ITD and the foliage table all match their sources to the precision those sources are published at. The problems are at the edges of the model, and two of them matter to a ray tracer:

1. **No surface in the crate is acoustically soft** (ACU-1). Every solid preset reflects more than 99.7% of the incident pressure, a few hundredths of a decibel per bounce, including `dry_vegetation` ("grass, scrub, thin brush"), which ISO 9613-2 classes as porous ground. The material a ray bounces off cannot change how a reverb or a ground reflection sounds. The module doc claimed the opposite. *Doc corrected; model open.*
2. **A source dropping behind a ridge steps by 5 dB at every frequency at once** (ACU-2). The barrier model has no lit-side branch. *Open.*

| ID | Sev | Finding | Status | Test |
|---|---|---|---|---|
| ACU-1 | High | Surface absorption comes only from the bulk impedance mismatch, so every solid reflects more than 99.7% of pressure (max 0.026 dB per bounce, `dry_vegetation`), and the model has no porous absorption. Measured absorption of grass-covered ground is tenths. The doc said "concrete rings and rubber does not" and "a sofa fixes it". | Doc fixed; **open** | `review_every_solid_preset_reflects_more_than_99_7_percent`, `review_vegetation_ground_is_acoustically_soft` (ignored) |
| ACU-2 | Medium | `barrier_insertion_db` is 0 dB for δ ≤ 0 and 5 dB for δ → 0⁺. Maekawa's curve continues into the lit side (Kurze–Anderson: 5 dB at N → 0⁻, reaching 0 near N = −0.19). A 5 dB broadband step is audible. | **Open** | `review_barrier_is_continuous_across_the_shadow_boundary` (ignored) |
| ACU-3 | Low | `impedance` used the thin-rod speed `sqrt(E/ρ)`, while its doc (correctly) said a thick slab needs the bulk longitudinal speed. That read steel 14% low, rubber 4.1× low and napalm 13× low (napalm R 0.674, now 0.970). It now uses the P-wave modulus `E(1−ν)/((1+ν)(1−2ν))`. | **Fixed** | `review_impedance_is_the_bulk_p_wave_impedance`, `review_an_incompressible_material_reflects_everything` |
| ACU-4 | Low | `doppler_ratio` clamped the source below Mach 1 but not a receding listener: −0.16 at 400 m/s, and exactly 0 at `c`. A negative playback rate reaches a resampler. The listener is now clamped the same way. | **Fixed** | `review_doppler_is_never_negative_for_a_receding_listener` |
| ACU-5 | Low | Non-finite inputs became NaN. `barrier_insertion_db(δ, wavelength(0 Hz))` was 0/0, because `wavelength` returns ∞ at 0 Hz by design. `absorption_db_per_m(NaN)` and `gain_from_db_loss(NaN)` also passed NaN on, and a NaN in a recursive filter stays there. All of them now return a number: the N → 0 limit of 5 dB, or no loss. | **Fixed** | `review_non_finite_inputs_never_become_nan` |
| ACU-6 | Low | The `speed_of_sound` doc said humidity adds "about 0.3 m/s at full saturation". The code adds 1.27 m/s at 20 °C, which agrees with Cramer (1.25). The code was right; the doc is fixed. | **Fixed** (doc) | `review_humidity_raises_the_speed_of_sound_by_cramers_amount` |
| ACU-P1 | Perf | `absorption_gain` costs 98 ns per call and `absorption_db_per_m` 81 ns: about nine transcendentals, none of which depends on frequency (it enters only through `f²` and two divisions). A per-band table is 19.5 ns (5×). `delay` costs 34 ns because `speed_of_sound` re-derives the saturation vapour pressure (two `powf`) every call. At 10k paths × 8 bands, recomputing every frame costs 7.8 ms. | Open (caller) | — |

---

## ACU-1: no surface is soft

`reflection_coefficient` is `|Z − Z_air| / (Z + Z_air)`, with `Z` the material's bulk impedance. For a solid, `Z` is 10⁵–10⁷ rayl against air's 413, so every preset reflects almost everything:

| Preset | Z (rayl) | R (pressure) | α (energy) | Loss per bounce |
|---|---:|---:|---:|---:|
| concrete | 8.9e6 | 0.99991 | 0.0002 | 0.0008 dB |
| ice | 3.6e6 | 0.99977 | 0.0005 | 0.002 dB |
| wood | 3.4e6 | 0.99975 | 0.0005 | 0.002 dB |
| rubber | 4.3e5 | 0.99810 | 0.004 | 0.017 dB |
| polyurethane | 3.0e5 | 0.99727 | 0.005 | 0.024 dB |
| dry_vegetation | 2.8e5 | 0.99703 | 0.006 | 0.026 dB |

For a thick solid slab that is the right answer. Concrete's α ≈ 0.01–0.02 in published tables comes from surface porosity this model does not see. At oblique incidence the answer only gets harder: beyond the critical angle `asin(c_air / c_L)` (3.4° for steel, 5.3° for concrete) the transmitted wave is evanescent and reflection is total. So normal incidence is the worst case, and the doc's "glancing angles reflect more" stands.

What makes grass, snow, leaf litter or a sofa absorb is air pumped through pores. It is governed by **flow resistivity** σ (Pa·s/m²), and `Material` carries no such number. ISO 9613-2 §7.3.1 draws the line explicitly: ground "covered by grass, trees or other vegetation" is porous (G = 1), and paving, water, ice and concrete are hard (G = 0). In this model `dry_vegetation` reflects like concrete to within 0.03 dB.

**For a ray tracer, that means the surface material has no audible effect.** Twenty bounces off scrub cost 0.5 dB in total. Every reverb tail and every ground reflection is shaped by geometry and air absorption alone.

**Proposed fix** (an API decision, so not made here):

- Give porous surfaces a flow resistivity, as a `Material` field or a separate ground-class enum.
- Derive a frequency-dependent surface impedance from it with Miki's (1990) form of Delany–Bazley.
- Reflect with the locally reacting plane-wave coefficient `R(θ) = (ζ cos θ − 1)/(ζ cos θ + 1)`.

The Nordtest ground classes supply the numbers without anyone tuning by ear: A (snow, moss) 12.5 kPa·s/m²; C (grass) 80; D (pasture) 200; E (compacted field, gravel) 500; G (asphalt, concrete) 20 000. That makes snow on a winter map soft and a road hard, from one number per surface. `AIR_IMPEDANCE` should come from `Air` at the same time (see Notes).

## ACU-2: the barrier step

`barrier_insertion_db` is the Kurze–Anderson approximation to Maekawa for a blocked path: `5 + 20 log10(√(2πN) / tanh √(2πN))`, N = 2δ/λ. It is right for N > 0: 13.1 dB at N = 1 against about 13 on Maekawa's chart.

It starts at 5 dB. A source exactly on the shadow boundary is already half-shadowed, and Maekawa's measured curve rises smoothly to that 5 dB from the lit side over about N ∈ (−0.2, 0). Kurze–Anderson's lit-side branch is `5 + 20 log10(√(2π|N|) / tan √(2π|N|))`. The function returns 0 for every δ ≤ 0, so a listener walking into a ridge's shadow gets a 5 dB step at every frequency in one frame.

The fix needs a signed path difference (negative when the line of sight clears the edge), and a caller that looks for diffracting edges near a clear line of sight as well as behind one. Both ends change, so it is left for a decision alongside the ray tracer. Changing only this function would give 5 dB to any caller that passes `0.0` to mean "clear".

---

## Verified correct

| What | Oracle | Result |
|---|---|---|
| ISO 9613-1 atmospheric absorption | Independent implementation of ISO 9613-1 eqs. (3)–(5) and Annex B | Agrees to 1e-15 relative, including off-standard pressure (80 kPa) |
| The same, at published values | ISO 9613-2 Table 2: six climates × eight octave bands at exact midband frequencies | All 48 within the table's rounding. The module's own test allowed ±40% at 1 kHz; the new one allows 0.06 dB/km or 0.5%. |
| Speed of sound | Ideal gas; Cramer (1993) for humidity | 343.29 m/s dry at 20 °C; humidity term within 0.02 m/s of Cramer at 20 °C and 30 °C |
| Water-vapour mole fraction | ISO 9613-1 Annex B | Same formula, same triple-point reference |
| Interaural time difference | Woodworth `(a/c)(θ + sin θ)` in interaural-polar coordinates | 0.654 ms at 90°. Uses `asin(right · û)`, so it is correct for elevated sources too. |
| Spreading, delay, dB ↔ gain | 1/r pressure law; definitions | Exact |
| Barrier, blocked side | Maekawa's chart; Kurze–Anderson | 13.1 dB at N = 1; saturates at 24 dB (ISO 9613-2 caps 20 single-edge, 25 multiple) |
| Foliage | ISO 9613-2 Annex A dense-foliage rates | Exact at the tabulated octaves; log-frequency interpolation between them |
| Reflection and absorption | Normal-incidence plane wave; energy balance | Correct formula; R² + α = 1 |

## Notes (modelling choices, not defects)

- **ITD is Woodworth's high-frequency limit.** Kuhn (1977) measured low-frequency ITDs larger than Woodworth's, up to about 1.5× near the median plane and about 0.76 ms against 0.65 ms at the side. `a_head_casts_no_shadow_at_low_frequencies` asserts frequency-independence by design, and that is a defensible simplification for a game.
- **`AIR_IMPEDANCE` is fixed at 20 °C** (413 rayl), while `Air` knows the weather: winter air (270 K) is 430.5 (+4%). Against solids that moves R in the sixth decimal. Against porous ground (ACU-1) it would matter, so the two should land together.
- **Foliage is continuous below 20 m.** ISO 9613-2 gives nothing below 10 m and a flat total for 10–20 m. The per-metre rate at every length avoids two steps a moving listener would hear, so it is kept.
- **`Ears` assumes a right-handed basis.** Up is `right × forward`. In a left-handed basis every elevation comes out mirrored. This is now documented on `Ears`.

## What to check first in the ray tracer

These are the failure modes the primitives above cannot prevent on their own, in rough order of how often they occur in image-source and ray-traced audio:

1. **Spreading on a reflected path uses the unfolded total length.** `spreading_gain(r1 + r2 + …)` is right. The product `spreading_gain(r1) · spreading_gain(r2)` falls as 1/r², which is the inverse-square mistake this module's doc warns about. The 1 m clamp would also apply per leg.
2. **Per-bounce loss is applied to the right quantity.** `reflection_coefficient` is a *pressure* ratio: multiply amplitude by R, or energy by R² = 1 − α. Multiplying energy by R, or amplitude by 1 − α, is wrong by a square.
3. **Absorption per path per band per frame** (ACU-P1). `Air` changes with the weather, not per frame, so a per-band table built on change makes the per-path cost one multiply.
4. **How it calls `barrier_insertion_db`** (ACU-2): whether it passes 0 for a clear line of sight, and whether it ever looks for an edge when the line of sight is clear.
5. **Delay from the total path length**, with one `speed_of_sound()` per frame rather than per path.
6. **Surface materials** (ACU-1): what it maps terrain, snow and vegetation to, since none of them will absorb.

## Validation

| Command | Result |
|---|---|
| `cargo test --features all` (CI) | lib 1102 passed, 0 failed, 34 ignored; doctests 237 passed |
| `cargo test` (default features, CI) | lib 651 passed, 0 failed, 18 ignored; doctests 135 passed |
| `cargo test --lib acoustics` | 36 passed, 2 ignored |
| `cargo test --lib acoustics -- --ignored` | both open repros fail as expected: a 5.00 dB step at 125 Hz, and 0.0059 energy absorbed by `dry_vegetation` |
| Regression tests against `993f35b`'s `acoustics/` | the 3 controls pass; the 5 fix tests fail |

Timings are from one release build on this container (x86_64), 2M calls each: `absorption_db_per_m` 81.0 ns, `absorption_gain` 97.6 ns, cached per-band gain 19.5 ns, `delay` 34.1 ns, `Ears::hear` 79.8 ns.
