//! Tests for the sparse-tile film step and the porous drink it calls.

use super::*;
use crate::fluid_dynamics::{infiltration, soak_uptake};

/// Blood in a 4 mm pool on a 0.2 m grid, the units Ridgeline's stain uses: display depth
/// 0..255 over 4 mm, 65 536 sub-units a display unit, falls of 0.8 mm.
fn blood() -> TileFilmLaw {
    const RHO: f32 = 1060.0;
    const G: f32 = 9.81;
    const MU: f32 = 0.004;
    const POOL_M: f32 = 0.004;
    const TEXEL_M: f32 = 0.2;
    const GRADE: f32 = 0.10;
    const DT: f32 = 1.0 / 60.0;
    let run_speed = RHO * G * GRADE * POOL_M * POOL_M / (3.0 * MU);
    let yield_length = 0.005 / (RHO * G);
    TileFilmLaw {
        courant: run_speed * DT / TEXEL_M,
        grade_per_head: (POOL_M / 255.0) / TEXEL_M / GRADE,
        drive_gain: GRADE * POOL_M,
        yield_length,
        yield_head: yield_length * TEXEL_M / ((POOL_M / 255.0) * (POOL_M / 255.0)),
        film: 24,
        subunits: 65_536,
        fall_gain: (0.0008 / (POOL_M / 255.0)) as i32 as f32,
        legacy_cohesion: None,
    }
}

/// A plain depth field: one `i32` a texel, earth-like retention, a porous bed that can
/// be switched off.
struct Field {
    width: usize,
    depth: Vec<i32>,
    held: Vec<u8>,
    porous: bool,
    drunk: i64,
}

impl Field {
    fn new(width: usize, height: usize) -> Field {
        Field { width, depth: vec![0; width * height], held: vec![0; width * height], porous: false, drunk: 0 }
    }
    fn total(&self) -> i64 {
        self.depth.iter().map(|&d| d as i64).sum()
    }
}

impl FilmField for Field {
    fn depth(&self, cell: usize) -> i32 {
        self.depth[cell]
    }
    fn set_depth(&mut self, cell: usize, depth: i32) {
        self.depth[cell] = depth.clamp(0, 255 * 65_536 + 65_535);
    }
    fn display(&self, cell: usize) -> u8 {
        (self.depth[cell] >> 16) as u8
    }
    fn row_wet(&self, y: usize, x0: usize, x1: usize) -> bool {
        self.depth[y * self.width + x0..y * self.width + x1].iter().any(|&d| d >> 16 != 0)
    }
    fn snapshot_row(&self, y: usize, x0: usize, x1: usize, surface: &mut [i16], wet: &mut [bool]) {
        for (i, x) in (x0..x1).enumerate() {
            let d = (self.depth[y * self.width + x] >> 16) as u8;
            surface[i] = d as i16;
            wet[i] = d > 24;
        }
    }
    fn retention(&self, _cell: usize) -> f32 {
        // Earth's: half the square of a 2.98 mm puddle in display units.
        let units = 2.983_352e-3 / (0.004 / 255.0);
        units * units * 0.5
    }
    fn thinning(&self, _x: usize, _y: usize, _depth: i32) -> f32 {
        1.0
    }
    fn may_dry(&self, _x: usize, _y: usize) -> bool {
        true
    }
    fn drink(&mut self, cell: usize, arrived: f32) -> i32 {
        if !self.porous {
            return 0;
        }
        let drunk = infiltration(arrived as i32, self.held[cell], 255, 235, 3);
        if drunk > 0 {
            let ink = ((drunk + 65_535) / 65_536) as u32;
            self.held[cell] = soak_uptake(self.held[cell], ink, 255);
            self.drunk += drunk as i64;
        }
        drunk
    }
}

fn pour(film: &mut TileFilm, field: &mut Field, x: usize, y: usize, display: u8) {
    field.depth[y * field.width + x] = (display as i32) << 16;
    film.wake(x, y);
}

/// A steady fall toward +x of `units` stored fall units a texel.
fn slope(width: usize, height: usize, units: i8) -> (Vec<i8>, Vec<i8>) {
    (vec![units; width * height], vec![0; width * height])
}

fn run(film: &mut TileFilm, field: &mut Field, law: &TileFilmLaw, fx: &[i8], fz: &[i8], n: usize) -> bool {
    let mut any = false;
    for i in 0..n {
        any |= film.step(field, law, fx, fz, i % 40 == 39).moved;
    }
    any
}

#[test]
fn a_dry_field_visits_no_tiles() {
    let (w, h) = (200, 150);
    let mut film = TileFilm::new(w, h);
    let mut field = Field::new(w, h);
    let (fx, fz) = slope(w, h, 10);
    let report = film.step(&mut field, &blood(), &fx, &fz, false);
    assert_eq!(report.tiles, 0);
    assert!(!report.moved);
    assert!(report.dirty_min.0 > report.dirty_max.0);
}

#[test]
fn without_a_porous_bed_the_flow_conserves_liquid_exactly() {
    let (w, h) = (96, 96);
    let mut film = TileFilm::new(w, h);
    let mut field = Field::new(w, h);
    let (fx, fz) = slope(w, h, 6);
    for (x, y) in [(20, 20), (31, 31), (32, 32), (60, 40)] {
        pour(&mut film, &mut field, x, y, 255);
    }
    let before = field.total();
    // No drying steps, so the only way liquid could change is the flow itself.
    for _ in 0..300 {
        film.step(&mut field, &blood(), &fx, &fz, false);
    }
    let after = field.total();
    // Each texel's arrival truncates to a whole sub-unit; that rounding is the only loss.
    let lost = before - after;
    assert!(lost >= 0 && (lost as f64) < before as f64 * 1e-3, "lost {lost} of {before}");
    assert!(field.depth.iter().filter(|&&d| d > 0).count() > 4, "the liquid never moved");
}

#[test]
fn liquid_runs_downhill_along_the_fall() {
    let (w, h) = (96, 40);
    let mut film = TileFilm::new(w, h);
    let mut field = Field::new(w, h);
    let (fx, fz) = slope(w, h, 25);
    pour(&mut film, &mut field, 20, 20, 255);
    run(&mut film, &mut field, &blood(), &fx, &fz, 120);
    let (mut east, mut west) = (0i64, 0i64);
    for y in 0..h {
        for x in 0..w {
            let d = field.depth[y * w + x] as i64;
            if x > 20 {
                east += d;
            } else if x < 20 {
                west += d;
            }
        }
    }
    assert!(east > 10 * west.max(1), "downhill {east} against uphill {west}");
}

#[test]
fn a_pool_spreads_at_the_same_rate_in_every_direction() {
    let n = 65;
    let mut film = TileFilm::new(n, n);
    let mut field = Field::new(n, n);
    let flat = vec![0i8; n * n];
    for dy in 0..3 {
        for dx in 0..3 {
            pour(&mut film, &mut field, 31 + dx, 31 + dy, 255);
        }
    }
    for _ in 0..200 {
        film.step(&mut field, &blood(), &flat, &flat, false);
    }
    let at = |x: usize, y: usize| field.depth[y * n + x];
    for r in 1..12 {
        let e = at(32 + r, 32);
        assert_eq!(e, at(32 - r, 32), "east and west differ at {r}");
        assert_eq!(e, at(32, 32 + r), "east and south differ at {r}");
        assert_eq!(e, at(32, 32 - r), "east and north differ at {r}");
    }
}

#[test]
fn a_running_film_soaks_into_the_bed_and_the_books_balance() {
    let (w, h) = (96, 40);
    let mut film = TileFilm::new(w, h);
    let mut field = Field::new(w, h);
    field.porous = true;
    let (fx, fz) = slope(w, h, 25);
    pour(&mut film, &mut field, 10, 20, 255);
    let before = field.total();
    let mut soaks = 0;
    for _ in 0..200 {
        soaks += film.step(&mut field, &blood(), &fx, &fz, false).soaks;
    }
    assert!(soaks > 0 && field.drunk > 0, "the bed drank nothing");
    let lost = before - field.total() - field.drunk;
    assert!(lost >= 0 && (lost as f64) < before as f64 * 1e-3, "unaccounted {lost}");
    assert!(field.held.iter().filter(|&&s| s > 0).count() > 3, "the trail left no stain");
}

#[test]
fn the_liquid_comes_to_rest_and_the_tiles_go_to_sleep() {
    let (w, h) = (128, 128);
    let mut film = TileFilm::new(w, h);
    let mut field = Field::new(w, h);
    let flat = vec![0i8; w * h];
    pour(&mut film, &mut field, 64, 64, 255);
    pour(&mut film, &mut field, 65, 64, 255);
    run(&mut film, &mut field, &blood(), &flat, &flat, 60 * 120);
    // Rested: a further minute changes no display byte and leaves nothing awake.
    let moved = run(&mut film, &mut field, &blood(), &flat, &flat, 60 * 60);
    assert!(!moved, "the picture was still changing after two minutes");
    assert!(!film.any_active(), "tiles still awake over liquid at rest");
}

#[test]
fn the_step_allocates_nothing_once_built() {
    let (w, h) = (128, 128);
    let mut film = TileFilm::new(w, h);
    let mut field = Field::new(w, h);
    let (fx, fz) = slope(w, h, 10);
    for x in 0..w {
        pour(&mut film, &mut field, x, 64, 200);
    }
    let capacity = film.live.capacity();
    let ptr = film.live.as_ptr();
    run(&mut film, &mut field, &blood(), &fx, &fz, 50);
    assert_eq!(film.live.capacity(), capacity);
    assert_eq!(film.live.as_ptr(), ptr, "the live list was reallocated");
}

#[test]
fn the_legacy_arm_holds_back_a_constant_head() {
    let mut law = blood();
    assert_eq!(law.threshold(0.5, 100.0, false), 50.0);
    assert_eq!(law.threshold(0.5, 100.0, true), law.yield_head * 0.5);
    law.legacy_cohesion = Some(24.0);
    assert_eq!(law.threshold(0.5, 100.0, false), 24.0);
    assert_eq!(law.threshold(9.0, 1.0, true), 24.0);
}

#[test]
fn outflow_stops_at_the_yield_and_never_passes_the_courant_cap() {
    let law = blood();
    let full = 255 * law.subunits;
    assert_eq!(law.outflow(0.0, full), None);
    let steep = law.outflow(1.0e6, full).unwrap();
    assert!((steep - law.courant).abs() < 1e-9, "{steep} vs {}", law.courant);
    let gentle = law.outflow(50.0, full).unwrap();
    assert!(gentle > 0.0 && gentle < law.courant);
    // The cube: a thinner film on the same slope moves a much smaller share.
    if let Some(thin) = law.outflow(50.0, full / 2) {
        assert!(thin < gentle / 3.0);
    }
}
