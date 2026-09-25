//! The records that cross the boundary: what the caller packs, what the GPU reads back, and
//! the limits and errors around them.
//!
//! Every record is `#[repr(C)]` and `Pod`, and its size is asserted at compile time,
//! because the WGSL reads the same bytes by word offset and a field that moved would be
//! read as another field rather than fail.

use bytemuck::{Pod, Zeroable};

use crate::acoustics::{Air, band, surfaces};

/// The four bands the laws are evaluated in, in hertz: octaves two apart from the bass to
/// the top of the treble. The fit (see [`crate::acoustics::band`]) turns them into one gain
/// and one low-pass per source.
pub const BANDS_HZ: [f32; 4] = [125.0, 500.0, 2_000.0, 8_000.0];

/// The probe band, in hertz: the one [`SourceResult::excess_m`] and the `blocked` and
/// `diffracted` flags report. 4 kHz, the game's `PROBE_HZ`, where a crack and a thump part
/// company behind cover.
pub const PROBE_HZ: f32 = 4_000.0;

/// Sources one dispatch may carry: the hard limit. A readback slot always holds this many
/// results.
pub const MAX_SOURCES: u32 = 128;

/// Large movers one dispatch may carry: the hard limit, one per lane of a source workgroup.
pub const MAX_MOVERS: u32 = 64;

/// Static obstacles a scene may hold: the hard limit.
pub const MAX_STATICS: u32 = 4_096;

/// The largest terrain side, in cells: the hard limit on both `cols` and `rows`.
pub const MAX_TERRAIN_SIDE: u32 = 1_024;

/// Directions the listener field traces: the hard limit, one per lane of its workgroup.
pub const MAX_FIELD_RAYS: u32 = 64;

/// Reflections a field ray follows: the hard limit.
pub const MAX_BOUNCES: u32 = 2;

/// Cells one static obstacle may cover in the scene grid: the hard limit. 4,096 cells is a
/// 128 m square at 2 m, larger than any building; the bound keeps the grid build's loop
/// constant.
pub const MAX_CELLS_PER_STATIC: u32 = 4_096;

/// Points a legibility curve may have.
pub const MAX_LEGIBILITY_POINTS: usize = 8;

/// The `ignore_mover` value that names no mover.
pub const NO_MOVER: u8 = 255;

/// The largest tag a [`Source`] can carry: 24 bits.
pub const MAX_TAG: u32 = (1 << 24) - 1;

/// Bytes of one readback slot: [`MAX_SOURCES`] results and the listener field.
pub const READBACK_BYTES: u64 =
    MAX_SOURCES as u64 * std::mem::size_of::<SourceResult>() as u64
        + std::mem::size_of::<ListenerField>() as u64;

/// Bytes of the dispatch header at the front of every packed dispatch.
pub const HEADER_BYTES: u64 = std::mem::size_of::<DispatchHeader>() as u64;

/// A sound source for one dispatch.
///
/// The position must be above the terrain column it stands in: the terrain is a column per
/// cell at the cell's height, and a source inside one is buried in it. A unit's origin at
/// mid-height is the right point.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Source {
    /// World position, in metres.
    pub position: [f32; 3],
    /// Gain toward the listener, 1 for a source that radiates evenly.
    pub directivity: f32,
    /// World velocity, in metres per second.
    pub velocity: [f32; 3],
    /// Bits 0 to 23: the tag echoed in the result. Bits 24 to 31: the index of the mover
    /// the source sits inside, or [`NO_MOVER`]. Build it with [`Source::new`].
    pub word: u32,
}

const _: () = assert!(std::mem::size_of::<Source>() == 32);

impl Source {
    /// A source, with its tag and the mover it sits inside.
    ///
    /// # Arguments
    ///
    /// * `position` - world position, in metres.
    /// * `directivity` - gain toward the listener; 1 for an even radiator.
    /// * `velocity` - world velocity, in metres per second.
    /// * `tag` - up to 24 bits, echoed in the [`SourceResult`] so the caller can tell a
    ///   result for this sound from one for whatever takes its slot later.
    /// * `ignore_mover` - the index into the dispatch's movers of the one the source is
    ///   inside (a vehicle's own engine), or [`NO_MOVER`].
    ///
    /// # Returns
    ///
    /// The packed record.
    ///
    /// # Errors
    ///
    /// [`AcousticsError::TagTooWide`] when `tag` exceeds [`MAX_TAG`].
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{Source, NO_MOVER};
    ///
    /// let s = Source::new([10.0, 1.0, -4.0], 1.0, [0.0; 3], 77, NO_MOVER).unwrap();
    /// assert_eq!(s.tag(), 77);
    /// assert_eq!(s.ignore_mover(), NO_MOVER);
    /// assert!(Source::new([0.0; 3], 1.0, [0.0; 3], 1 << 24, NO_MOVER).is_err());
    /// ```
    pub fn new(
        position: [f32; 3],
        directivity: f32,
        velocity: [f32; 3],
        tag: u32,
        ignore_mover: u8,
    ) -> Result<Source, AcousticsError> {
        if tag > MAX_TAG {
            return Err(AcousticsError::TagTooWide { tag });
        }
        Ok(Source {
            position,
            directivity,
            velocity,
            word: tag | (ignore_mover as u32) << 24,
        })
    }

    /// The 24-bit tag.
    ///
    /// # Returns
    ///
    /// Bits 0 to 23 of the word.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{Source, NO_MOVER};
    ///
    /// assert_eq!(Source::new([0.0; 3], 1.0, [0.0; 3], 5, 3).unwrap().tag(), 5);
    /// ```
    pub fn tag(&self) -> u32 {
        self.word & MAX_TAG
    }

    /// The mover this source sits inside, or [`NO_MOVER`].
    ///
    /// # Returns
    ///
    /// Bits 24 to 31 of the word.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::Source;
    ///
    /// assert_eq!(Source::new([0.0; 3], 1.0, [0.0; 3], 5, 3).unwrap().ignore_mover(), 3);
    /// ```
    pub fn ignore_mover(&self) -> u8 {
        (self.word >> 24) as u8
    }
}

/// Who is listening, where, facing which way and moving how fast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener {
    /// World position of the ears, in metres.
    pub position: [f32; 3],
    /// Unit vector the listener faces.
    pub forward: [f32; 3],
    /// Unit vector to the listener's right; perpendicular to `forward`.
    pub right: [f32; 3],
    /// World velocity, in metres per second.
    pub velocity: [f32; 3],
}

/// The per-dispatch header: the listener, the air, the legibility curve and the counts.
///
/// 160 bytes: four 16-byte vectors for the listener (the speed of sound rides in the
/// position's `w`), the four bands' air absorption, eight `(metres, hertz)` points and the
/// counts. Build it with [`DispatchHeader::new`]; [`crate::gpu::acoustics::GpuAcoustics::pack_dispatch`]
/// fills the counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct DispatchHeader {
    /// Listener position in metres, and the speed of sound in metres per second in `w`.
    pub listener: [f32; 4],
    /// Listener forward, and a pad.
    pub forward: [f32; 4],
    /// Listener right, and a pad.
    pub right: [f32; 4],
    /// Listener velocity in metres per second, and a pad.
    pub velocity: [f32; 4],
    /// Air absorption in each of [`BANDS_HZ`], in decibels per metre.
    pub air_db_per_m: [f32; 4],
    /// The legibility curve: up to eight `(metres, hertz)` points, ascending in distance.
    pub legibility: [[f32; 2]; MAX_LEGIBILITY_POINTS],
    /// Sources, movers, legibility points, and a pad.
    pub counts: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<DispatchHeader>() == 160);

impl DispatchHeader {
    /// A header for a listener in an air, with an optional legibility curve.
    ///
    /// The air's absorption is evaluated at each band here, on the CPU in f64 (ISO 9613-1),
    /// and shipped as f32; the GPU multiplies it by the distance.
    ///
    /// # Arguments
    ///
    /// * `listener` - position, orientation and velocity.
    /// * `air` - the air the sound crosses; its speed of sound and absorption.
    /// * `legibility` - `(distance_m, cutoff_hz)` points, ascending in distance, capping the
    ///   cutoff at distance so a far sound stays legible as far; empty for none.
    ///
    /// # Returns
    ///
    /// The header, with zero counts.
    ///
    /// # Errors
    ///
    /// [`AcousticsError::Legibility`] when the curve has more than
    /// [`MAX_LEGIBILITY_POINTS`] points, is not ascending in distance, or holds a
    /// non-finite or non-positive cutoff.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::acoustics::Air;
    /// use rs_physics::gpu::acoustics::{DispatchHeader, Listener};
    ///
    /// let listener = Listener {
    ///     position: [0.0, 1.6, 0.0],
    ///     forward: [0.0, 0.0, -1.0],
    ///     right: [1.0, 0.0, 0.0],
    ///     velocity: [0.0; 3],
    /// };
    /// let ladder = [(0.0, 20_000.0), (45.0, 1_400.0), (130.0, 480.0)];
    /// let h = DispatchHeader::new(&listener, &Air::standard(), &ladder).unwrap();
    /// assert!((h.listener[3] - 343.0).abs() < 2.0);
    /// assert_eq!(h.counts[2], 3);
    /// ```
    pub fn new(
        listener: &Listener,
        air: &Air,
        legibility: &[(f32, f32)],
    ) -> Result<DispatchHeader, AcousticsError> {
        if legibility.len() > MAX_LEGIBILITY_POINTS {
            return Err(AcousticsError::Legibility("more than eight points"));
        }
        let mut points = [[0.0f32; 2]; MAX_LEGIBILITY_POINTS];
        let mut last = f32::NEG_INFINITY;
        for (i, &(m, hz)) in legibility.iter().enumerate() {
            if !(m >= last) || !m.is_finite() {
                return Err(AcousticsError::Legibility("distances must ascend"));
            }
            if !(hz > 0.0) || !hz.is_finite() {
                return Err(AcousticsError::Legibility("cutoffs must be positive"));
            }
            points[i] = [m, hz];
            last = m;
        }
        let c = air.speed_of_sound() as f32;
        let p = listener.position;
        let (f, r, v) = (listener.forward, listener.right, listener.velocity);
        Ok(DispatchHeader {
            listener: [p[0], p[1], p[2], c],
            forward: [f[0], f[1], f[2], 0.0],
            right: [r[0], r[1], r[2], 0.0],
            velocity: [v[0], v[1], v[2], 0.0],
            air_db_per_m: BANDS_HZ.map(|hz| air.absorption_db_per_m(hz as f64) as f32),
            legibility: points,
            counts: [0, 0, legibility.len() as u32, 0],
        })
    }
}

/// An oriented box: the engine's instance convention, rows 0 to 2 of the 3x4 transform of
/// the origin-centred unit cube (corners at +-0.5), then the acoustic material and the
/// half-width the Fresnel rule tests.
///
/// The acoustics treat it as an upright column over its footprint: a path is obstructed
/// where it passes below the box's top and above its bottom, and the excess is measured
/// over the top edge. Tilt beyond yaw is honoured in the extents, not in the edge.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Obb {
    /// Rows 0 to 2 of the unit cube's local-to-world transform: `[m00, m01, m02, tx]` and
    /// so on, in metres.
    pub rows: [[f32; 4]; 3],
    /// Index into the scene's [`AcousticMaterial`] table (statics; ignored for movers).
    pub material: u32,
    /// Lateral half-width across a path, in metres: what the Fresnel rule compares with the
    /// first-zone radius. The narrower horizontal half-extent is the honest figure.
    pub half_width: f32,
    /// Padding to 64 bytes.
    pub pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<Obb>() == 64);

impl Obb {
    /// An upright box: a centre, full sizes along its own axes, and a yaw.
    ///
    /// # Arguments
    ///
    /// * `center` - world centre, in metres.
    /// * `size` - full extents along the box's local x, y and z, in metres.
    /// * `yaw_rad` - rotation about world +y, in radians.
    /// * `material` - index into the scene's material table.
    ///
    /// # Returns
    ///
    /// The box, with `half_width` the smaller horizontal half-extent.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::Obb;
    ///
    /// let wall = Obb::upright([0.0, 2.0, 5.0], [8.0, 4.0, 0.4], 0.0, 0);
    /// assert_eq!(wall.rows[1][3], 2.0);
    /// assert!((wall.half_width - 0.2).abs() < 1e-6);
    /// ```
    pub fn upright(center: [f32; 3], size: [f32; 3], yaw_rad: f32, material: u32) -> Obb {
        let (s, c) = yaw_rad.sin_cos();
        // Columns are the local axes scaled by the size: x' = (c, 0, -s), z' = (s, 0, c).
        Obb {
            rows: [
                [c * size[0], 0.0, s * size[2], center[0]],
                [0.0, size[1], 0.0, center[1]],
                [-s * size[0], 0.0, c * size[2], center[2]],
            ],
            material,
            half_width: 0.5 * size[0].min(size[2]),
            pad: [0; 2],
        }
    }
}

/// A surface's acoustics: the pressure reflection coefficient in each of [`BANDS_HZ`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct AcousticMaterial {
    /// Pressure reflection coefficient per band, `0..=1`.
    pub reflection: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<AcousticMaterial>() == 16);

impl AcousticMaterial {
    /// A material's acoustics from its impedance: the same coefficient in every band.
    ///
    /// # Arguments
    ///
    /// * `material` - a physical material; see [`surfaces::reflection_coefficient`].
    ///
    /// # Returns
    ///
    /// The four-band record.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::AcousticMaterial;
    /// use rs_physics::materials::Material;
    ///
    /// let concrete = AcousticMaterial::from_material(&Material::concrete());
    /// assert!(concrete.reflection[0] > 0.99);
    /// ```
    pub fn from_material(material: &crate::materials::Material) -> AcousticMaterial {
        let r = surfaces::reflection_coefficient(material) as f32;
        AcousticMaterial { reflection: [r; 4] }
    }
}

/// One source's result.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SourceResult {
    /// Amplitude gain at the left ear: spreading, the fitted broadband gain and the head
    /// shadow at the probe band.
    pub gain_l: f32,
    /// Amplitude gain at the right ear.
    pub gain_r: f32,
    /// Interaural delay in seconds, signed: positive when the right ear leads.
    pub itd_s: f32,
    /// One-pole low-pass cutoff, in hertz.
    pub cutoff_hz: f32,
    /// Doppler pitch ratio; 1 is no shift.
    pub pitch: f32,
    /// The main edge's signed excess path at the probe band, in metres: positive in the
    /// shadow, negative in the lit zone, `f32::MIN` when nothing lies under the path.
    pub excess_m: f32,
    /// Path length through foliage, in metres.
    pub foliage_m: f32,
    /// Bits 0 to 23: the source's tag. Bits 24 to 31: [`SourceResult::BLOCKED`],
    /// [`SourceResult::DIFFRACTED`], [`SourceResult::LEGIBILITY_CLAMPED`].
    pub word: u32,
}

const _: () = assert!(std::mem::size_of::<SourceResult>() == 32);

impl SourceResult {
    /// Flag: the line of sight is cut at the probe band (`excess_m > 0`).
    pub const BLOCKED: u8 = 1;
    /// Flag: the probe band loses something to an edge (the path is inside the lit zone
    /// of one, or in its shadow).
    pub const DIFFRACTED: u8 = 2;
    /// Flag: the legibility curve lowered the cutoff.
    pub const LEGIBILITY_CLAMPED: u8 = 4;

    /// The tag of the source this result is for.
    ///
    /// # Returns
    ///
    /// Bits 0 to 23 of the word.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::SourceResult;
    ///
    /// let r = SourceResult { word: 42 | (1 << 24), ..Default::default() };
    /// assert_eq!(r.tag(), 42);
    /// assert!(r.blocked());
    /// ```
    pub fn tag(&self) -> u32 {
        self.word & MAX_TAG
    }

    /// The flag byte.
    ///
    /// # Returns
    ///
    /// Bits 24 to 31 of the word.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::SourceResult;
    ///
    /// let r = SourceResult { word: 3 << 24, ..Default::default() };
    /// assert_eq!(r.flags(), SourceResult::BLOCKED | SourceResult::DIFFRACTED);
    /// ```
    pub fn flags(&self) -> u8 {
        (self.word >> 24) as u8
    }

    /// Whether the line of sight is cut at the probe band.
    ///
    /// # Returns
    ///
    /// `true` when [`SourceResult::BLOCKED`] is set.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::SourceResult;
    ///
    /// assert!(!SourceResult::default().blocked());
    /// ```
    pub fn blocked(&self) -> bool {
        self.flags() & Self::BLOCKED != 0
    }

    /// Whether an edge takes anything from the probe band.
    ///
    /// # Returns
    ///
    /// `true` when [`SourceResult::DIFFRACTED`] is set.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::SourceResult;
    ///
    /// assert!(!SourceResult::default().diffracted());
    /// ```
    pub fn diffracted(&self) -> bool {
        self.flags() & Self::DIFFRACTED != 0
    }

    /// Whether the legibility curve lowered the cutoff.
    ///
    /// # Returns
    ///
    /// `true` when [`SourceResult::LEGIBILITY_CLAMPED`] is set.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::SourceResult;
    ///
    /// assert!(!SourceResult::default().legibility_clamped());
    /// ```
    pub fn legibility_clamped(&self) -> bool {
        self.flags() & Self::LEGIBILITY_CLAMPED != 0
    }
}

/// One early-reflection tap.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Tap {
    /// Delay after the direct sound's emission at the listener, in seconds.
    pub delay_s: f32,
    /// Amplitude gain relative to the source's level at 1 m.
    pub gain: f32,
    /// Pan, -1 (left) to 1 (right), from the arrival direction against `right`.
    pub pan: f32,
    /// Padding.
    pub pad: f32,
}

/// The listener's surroundings: early reflections and the size of the space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ListenerField {
    /// Up to eight taps, the strongest arrival in each of the eight strongest 5 ms bins,
    /// ascending in delay; unused taps have zero gain.
    pub taps: [Tap; 8],
    /// Mean free path `4V / S` of the space the field rays hit, in metres (Kosten); 0 when
    /// no ray hits.
    pub mfp_m: f32,
    /// Eyring reverberation time at 500 Hz, in seconds; 0 in the open.
    pub rt60_s: f32,
    /// Share of the sphere, by solid angle, whose rays escape.
    pub clear_fraction: f32,
    /// Padding.
    pub pad: f32,
}

const _: () = assert!(std::mem::size_of::<ListenerField>() == 144);
const _: () = assert!(READBACK_BYTES == 4_240);

/// One field ray's path, for the oracles: where it hit and where it went.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FieldRay {
    /// Distance to the first hit, in metres; negative when the ray escaped.
    pub first_m: f32,
    /// Unit direction after the first reflection.
    pub first_dir: [f32; 3],
    /// Distance from the first hit to the second, in metres; negative when it escaped.
    pub second_m: f32,
    /// Unit direction after the second reflection.
    pub second_dir: [f32; 3],
    /// Cosine between the ray and the first surface's normal.
    pub first_cos: f32,
    /// Material index at the first hit.
    pub first_material: u32,
    /// The first reflection's arrival back at the listener: delay in seconds, gain, pan.
    pub first_arrival: [f32; 3],
    /// The second reflection's arrival: delay in seconds, gain, pan; zero when none.
    pub second_arrival: [f32; 3],
}

const _: () = assert!(std::mem::size_of::<FieldRay>() == 64);

/// One source's main edges as the march found them, for the oracles: the per-band maxima
/// that the laws were given.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SourceEdges {
    /// The largest signed excess path per band of [`BANDS_HZ`], in metres, with each
    /// obstacle admitted by the Fresnel rule at that band; `f32::MIN` for none.
    pub excess_m: [f32; 4],
    /// The same at [`PROBE_HZ`].
    pub probe_excess_m: f32,
    /// Path length through foliage, in metres.
    pub foliage_m: f32,
    /// Padding to 32 bytes.
    pub pad: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<SourceEdges>() == 32);

/// Everything one readback slot carries, copied out by
/// [`crate::gpu::acoustics::GpuAcoustics::take_ready`].
#[derive(Clone, Debug)]
pub struct TickResults {
    /// Results in dispatch order; the first `len` are valid.
    pub sources: [SourceResult; MAX_SOURCES as usize],
    /// How many of `sources` the dispatch carried.
    pub len: usize,
    /// The listener field.
    pub field: ListenerField,
    /// The dispatch's sequence number: 1 for the first dispatch encoded, and so on.
    pub sequence: u64,
}

impl Default for TickResults {
    fn default() -> TickResults {
        TickResults {
            sources: [SourceResult::default(); MAX_SOURCES as usize],
            len: 0,
            field: ListenerField::default(),
            sequence: 0,
        }
    }
}

impl TickResults {
    /// The valid results.
    ///
    /// # Returns
    ///
    /// `&sources[..len]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::TickResults;
    ///
    /// assert!(TickResults::default().results().is_empty());
    /// ```
    pub fn results(&self) -> &[SourceResult] {
        &self.sources[..self.len]
    }
}

/// The sizes a [`crate::gpu::acoustics::GpuAcoustics`] is built for. Each is at most its
/// hard limit ([`AcousticLimits::MAX`]); going over at `new` is an error, and so is a call
/// that asks for more later. Nothing is ever truncated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcousticLimits {
    /// Sources per dispatch.
    pub sources: u32,
    /// Movers per dispatch.
    pub movers: u32,
    /// Static obstacles per scene.
    pub statics: u32,
    /// Terrain columns.
    pub terrain_cols: u32,
    /// Terrain rows.
    pub terrain_rows: u32,
    /// Listener field rays.
    pub field_rays: u32,
    /// Field ray reflections.
    pub bounces: u32,
}

impl AcousticLimits {
    /// The hard limits: 128 sources, 64 movers, 4,096 statics, 1,024 x 1,024 cells,
    /// 64 field rays, 2 bounces.
    pub const MAX: AcousticLimits = AcousticLimits {
        sources: MAX_SOURCES,
        movers: MAX_MOVERS,
        statics: MAX_STATICS,
        terrain_cols: MAX_TERRAIN_SIDE,
        terrain_rows: MAX_TERRAIN_SIDE,
        field_rays: MAX_FIELD_RAYS,
        bounces: MAX_BOUNCES,
    };

    /// Check every limit against its hard limit.
    ///
    /// # Returns
    ///
    /// `Ok(())` when every field is within [`AcousticLimits::MAX`] and the field has its
    /// full ray set.
    ///
    /// # Errors
    ///
    /// [`AcousticsError::Limit`] naming the first field over its hard limit.
    /// The field ray count and bounce count are fixed by the shader: anything but the
    /// hard limit is refused too, because the direction table is derived for 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::AcousticLimits;
    ///
    /// assert!(AcousticLimits::MAX.validate().is_ok());
    /// let over = AcousticLimits { sources: 129, ..AcousticLimits::MAX };
    /// assert!(over.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), AcousticsError> {
        let m = AcousticLimits::MAX;
        let check = |what: &'static str, got: u32, max: u32| {
            if got > max {
                Err(AcousticsError::Limit { what, got: got as u64, max: max as u64 })
            } else {
                Ok(())
            }
        };
        check("sources", self.sources, m.sources)?;
        check("movers", self.movers, m.movers)?;
        check("statics", self.statics, m.statics)?;
        check("terrain columns", self.terrain_cols, m.terrain_cols)?;
        check("terrain rows", self.terrain_rows, m.terrain_rows)?;
        check("field rays", self.field_rays, m.field_rays)?;
        check("bounces", self.bounces, m.bounces)?;
        if self.field_rays != m.field_rays || self.bounces != m.bounces {
            return Err(AcousticsError::Limit {
                what: "field rays and bounces are fixed at 64 and 2",
                got: (self.field_rays as u64) << 32 | self.bounces as u64,
                max: (m.field_rays as u64) << 32 | m.bounces as u64,
            });
        }
        Ok(())
    }
}

impl Default for AcousticLimits {
    fn default() -> AcousticLimits {
        AcousticLimits::MAX
    }
}

/// Counts over the life of a [`crate::gpu::acoustics::GpuAcoustics`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AcousticCounters {
    /// Dispatches recorded.
    pub dispatches: u64,
    /// Dispatches skipped because their readback slot was still mapped or pending.
    pub skipped_busy: u64,
    /// Ready results unmapped unread because a newer one was taken.
    pub overwritten_ready: u64,
    /// Readback maps that failed (a lost device); the slot is reused.
    pub map_failures: u64,
}

/// What went wrong. Every limit is an error rather than a truncation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AcousticsError {
    /// A count over its limit.
    Limit {
        /// What was counted.
        what: &'static str,
        /// How many were asked for.
        got: u64,
        /// The limit.
        max: u64,
    },
    /// A tag wider than 24 bits.
    TagTooWide {
        /// The tag.
        tag: u32,
    },
    /// An output slice too small for what is packed into it.
    OutputTooSmall {
        /// Bytes needed.
        need: usize,
        /// Bytes given.
        got: usize,
    },
    /// Inputs whose shapes disagree (a grid's slice lengths, a rect outside the grid, a
    /// staged length that is not the packed length).
    Shape(&'static str),
    /// A material index past the end of the material table.
    Material {
        /// The index.
        index: u32,
        /// The table's length.
        len: u32,
    },
    /// A static obstacle outside the terrain grid, or covering more than
    /// [`MAX_CELLS_PER_STATIC`] cells.
    Static {
        /// The obstacle's index.
        index: u32,
        /// Why.
        why: &'static str,
    },
    /// A staged slice whose offset or length is not a multiple of 4 bytes.
    Misaligned,
    /// A malformed legibility curve.
    Legibility(&'static str),
    /// The adapter lacks something the pass needs.
    Device(&'static str),
}

impl std::fmt::Display for AcousticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcousticsError::Limit { what, got, max } => {
                write!(f, "{what}: {got} asked for, the limit is {max}")
            }
            AcousticsError::TagTooWide { tag } => write!(f, "tag {tag} is wider than 24 bits"),
            AcousticsError::OutputTooSmall { need, got } => {
                write!(f, "output needs {need} bytes, {got} given")
            }
            AcousticsError::Shape(why) => write!(f, "shape mismatch: {why}"),
            AcousticsError::Material { index, len } => {
                write!(f, "material {index} past the table's {len}")
            }
            AcousticsError::Static { index, why } => write!(f, "static {index}: {why}"),
            AcousticsError::Misaligned => write!(f, "staged offset or length not a multiple of 4"),
            AcousticsError::Legibility(why) => write!(f, "legibility curve: {why}"),
            AcousticsError::Device(why) => write!(f, "device: {why}"),
        }
    }
}

impl std::error::Error for AcousticsError {}

/// The CPU constants the shader is generated from, so none is typed twice.
pub(crate) fn law_constants() -> LawConstants {
    let x = BANDS_HZ.map(|f| {
        let r = f as f64 / band::FIT_REFERENCE_HZ;
        r * r
    });
    let mean = x.iter().sum::<f64>() / 4.0;
    let sxx: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    LawConstants {
        fit_x_mean: mean,
        fit_weights: x.map(|v| (v - mean) / sxx),
        foliage_db_per_m: BANDS_HZ.map(|f| surfaces::foliage_absorption_db_per_m(f as f64)),
    }
}

/// See [`law_constants`].
pub(crate) struct LawConstants {
    pub fit_x_mean: f64,
    pub fit_weights: [f64; 4],
    pub foliage_db_per_m: [f64; 4],
}
