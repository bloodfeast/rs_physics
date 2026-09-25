//! Packing: pure functions from the caller's data to the bytes the GPU reads.
//!
//! **Nothing here touches a device.** The caller writes these bytes into memory it owns
//! (the engine's staging ring), and hands [`GpuAcoustics`](super::GpuAcoustics) a
//! [`Staged`](super::Staged) slice to copy from while recording. None of these functions
//! allocates.
//!
//! # The scene buffer
//!
//! The scene is one storage buffer of 32-bit words, so the pass binds three buffers in all
//! and runs within wgpu's default of eight per stage. It opens with a header of
//! [`HEADER_WORDS`] words that holds the grid's shape and every region's word offset; the
//! regions follow in this order:
//!
//! | Region | Words | Written by |
//! |---|---|---|
//! | header | 48 | [`pack_scene`] |
//! | terrain heights, f32 per cell | cells | [`pack_scene`], [`pack_terrain_rect`] |
//! | terrain material, u8 per cell | cells / 4 | [`pack_scene`] |
//! | foliage density, u8 per cell | cells / 4, or none | [`pack_scene`] |
//! | statics, [`Obb`] | 16 each | [`pack_scene`] |
//! | statics' cell ranges | 4 each | [`pack_scene`] |
//! | materials, [`AcousticMaterial`] | 4 each | [`pack_scene`] |
//! | statics' inverse transforms | 12 each | the GPU |
//! | grid counts, heads and lists | cells, cells + 1, one per covered cell | the GPU |
//! | static tops per cell | cells | the GPU |
//! | max pyramid, levels 1 to 3 | about cells / 3 | the GPU |
//!
//! Only the first seven are staged; the rest are derived on the GPU by
//! [`GpuAcoustics::encode_scene`](super::GpuAcoustics::encode_scene).

use super::records::*;

/// Words in the scene header.
pub const HEADER_WORDS: u32 = 48;

/// Pyramid levels built above the terrain: level 3 is 8 x 8 cells, 16 m at 2 m cells,
/// the block the field march skips empty air by.
pub const PYRAMID_LEVELS: u32 = 3;

/// Metres by which a static's footprint is widened before it is binned into cells, so a
/// crossing point that rounds onto a cell edge still finds the static in that cell's list.
/// A millimetre is thousands of f32 ulps at map scale and invisible in the result.
const BIN_MARGIN_M: f64 = 1e-3;

/// A terrain: a column per cell at the cell's height.
///
/// Cell `(i, j)` covers `x` in `[origin[0] + i cell_m, origin[0] + (i + 1) cell_m)` and `z`
/// likewise from `origin[1]`, and sits at `heights[j * cols + i]`. The grid is also the
/// static obstacles' index: a static is seen where its footprint overlaps the grid.
#[derive(Clone, Copy, Debug)]
pub struct TerrainGrid<'a> {
    /// Column heights, row-major, in metres.
    pub heights: &'a [f32],
    /// Index into the material table per cell, row-major.
    pub material: &'a [u8],
    /// Cells along x.
    pub cols: u32,
    /// Cells along z.
    pub rows: u32,
    /// Cell side, in metres.
    pub cell_m: f32,
    /// World x and z of cell (0, 0)'s low corner, in metres.
    pub origin: [f32; 2],
}

/// Foliage density on the terrain's grid: 0 to 255 per cell mapping to 0 to 1 of foliage
/// per metre of path.
#[derive(Clone, Copy, Debug)]
pub struct DensityGrid<'a> {
    /// Density per cell, row-major, the terrain's shape.
    pub density: &'a [u8],
}

/// A rectangle of terrain cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridRect {
    /// First column.
    pub col: u32,
    /// First row.
    pub row: u32,
    /// Columns.
    pub cols: u32,
    /// Rows.
    pub rows: u32,
}

/// Where everything goes in the scene buffer, and how big it is.
///
/// Returned by [`GpuAcoustics::scene_bytes`](super::GpuAcoustics::scene_bytes); pass it to
/// [`pack_scene`] and to [`GpuAcoustics::encode_scene`](super::GpuAcoustics::encode_scene).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneLayout {
    pub(crate) header: [u32; HEADER_WORDS as usize],
    pub(crate) staged_words: u32,
    pub(crate) total_words: u32,
}

impl SceneLayout {
    /// Bytes [`pack_scene`] writes and the caller stages.
    ///
    /// # Returns
    ///
    /// The staged size, bytes; a multiple of 4.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{GpuAcoustics, TerrainGrid};
    ///
    /// let heights = [0.0f32; 4];
    /// let terrain = TerrainGrid {
    ///     heights: &heights, material: &[0; 4], cols: 2, rows: 2, cell_m: 2.0, origin: [0.0; 2],
    /// };
    /// let layout = GpuAcoustics::scene_bytes(&terrain, &[], 1, None).unwrap();
    /// assert!(layout.staged_bytes() > 0 && layout.staged_bytes() % 4 == 0);
    /// ```
    pub fn staged_bytes(&self) -> u64 {
        self.staged_words as u64 * 4
    }

    /// Bytes the scene occupies on the GPU, derived regions included.
    ///
    /// # Returns
    ///
    /// The resident size, bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{GpuAcoustics, TerrainGrid};
    ///
    /// let heights = [0.0f32; 4];
    /// let terrain = TerrainGrid {
    ///     heights: &heights, material: &[0; 4], cols: 2, rows: 2, cell_m: 2.0, origin: [0.0; 2],
    /// };
    /// let layout = GpuAcoustics::scene_bytes(&terrain, &[], 1, None).unwrap();
    /// assert!(layout.resident_bytes() > layout.staged_bytes());
    /// ```
    pub fn resident_bytes(&self) -> u64 {
        self.total_words as u64 * 4
    }

    /// Terrain columns and rows.
    ///
    /// # Returns
    ///
    /// `(cols, rows)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{GpuAcoustics, TerrainGrid};
    ///
    /// let heights = [0.0f32; 6];
    /// let terrain = TerrainGrid {
    ///     heights: &heights, material: &[0; 6], cols: 3, rows: 2, cell_m: 2.0, origin: [0.0; 2],
    /// };
    /// assert_eq!(GpuAcoustics::scene_bytes(&terrain, &[], 1, None).unwrap().grid(), (3, 2));
    /// ```
    pub fn grid(&self) -> (u32, u32) {
        (self.header[H_COLS], self.header[H_ROWS])
    }

    /// Static obstacles in the scene.
    ///
    /// # Returns
    ///
    /// The count.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::gpu::acoustics::{GpuAcoustics, TerrainGrid};
    ///
    /// let heights = [0.0f32; 4];
    /// let terrain = TerrainGrid {
    ///     heights: &heights, material: &[0; 4], cols: 2, rows: 2, cell_m: 2.0, origin: [0.0; 2],
    /// };
    /// assert_eq!(GpuAcoustics::scene_bytes(&terrain, &[], 1, None).unwrap().statics(), 0);
    /// ```
    pub fn statics(&self) -> u32 {
        self.header[H_STATICS]
    }

    pub(crate) fn words(&self, at: usize) -> u32 {
        self.header[at]
    }
}

// Header word indices. The WGSL declares the same list; `shader::source` checks they agree.
pub(crate) const H_COLS: usize = 0;
pub(crate) const H_ROWS: usize = 1;
pub(crate) const H_CELL: usize = 2;
pub(crate) const H_ORIGIN_X: usize = 3;
pub(crate) const H_ORIGIN_Z: usize = 4;
pub(crate) const H_STATICS: usize = 5;
pub(crate) const H_MATERIALS: usize = 6;
pub(crate) const H_HAS_FOLIAGE: usize = 7;
pub(crate) const H_OFF_HEIGHTS: usize = 8;
pub(crate) const H_OFF_MATERIAL: usize = 9;
pub(crate) const H_OFF_FOLIAGE: usize = 10;
pub(crate) const H_OFF_STATICS: usize = 11;
pub(crate) const H_OFF_RANGES: usize = 12;
pub(crate) const H_OFF_MATERIALS: usize = 13;
pub(crate) const H_OFF_INV: usize = 14;
pub(crate) const H_OFF_COUNT: usize = 15;
pub(crate) const H_OFF_HEADS: usize = 16;
pub(crate) const H_OFF_LIST: usize = 17;
pub(crate) const H_OFF_TOP: usize = 18;
pub(crate) const H_OFF_PYR: usize = 19; // three words, levels 1 to 3
pub(crate) const H_PYR_DIMS: usize = 22; // six words, (cols, rows) per level
pub(crate) const H_LIST_LEN: usize = 28;
pub(crate) const H_INV_CELL: usize = 29;
pub(crate) const H_RECT: usize = 30; // four words: col, row, cols, rows

/// Word offsets of the named header fields, for the WGSL generator.
pub(crate) const HEADER_FIELDS: &[(&str, usize)] = &[
    ("H_COLS", H_COLS),
    ("H_ROWS", H_ROWS),
    ("H_CELL", H_CELL),
    ("H_ORIGIN_X", H_ORIGIN_X),
    ("H_ORIGIN_Z", H_ORIGIN_Z),
    ("H_STATICS", H_STATICS),
    ("H_MATERIALS", H_MATERIALS),
    ("H_HAS_FOLIAGE", H_HAS_FOLIAGE),
    ("H_OFF_HEIGHTS", H_OFF_HEIGHTS),
    ("H_OFF_MATERIAL", H_OFF_MATERIAL),
    ("H_OFF_FOLIAGE", H_OFF_FOLIAGE),
    ("H_OFF_STATICS", H_OFF_STATICS),
    ("H_OFF_RANGES", H_OFF_RANGES),
    ("H_OFF_MATERIALS", H_OFF_MATERIALS),
    ("H_OFF_INV", H_OFF_INV),
    ("H_OFF_COUNT", H_OFF_COUNT),
    ("H_OFF_HEADS", H_OFF_HEADS),
    ("H_OFF_LIST", H_OFF_LIST),
    ("H_OFF_TOP", H_OFF_TOP),
    ("H_OFF_PYR", H_OFF_PYR),
    ("H_PYR_DIMS", H_PYR_DIMS),
    ("H_LIST_LEN", H_LIST_LEN),
    ("H_INV_CELL", H_INV_CELL),
    ("H_RECT", H_RECT),
];

/// The cells a static covers: its world AABB, widened by [`BIN_MARGIN_M`], clipped to the
/// grid. `Ok(None)` is impossible; a static entirely outside is an error.
pub(crate) fn static_range(
    index: u32,
    obb: &Obb,
    cols: u32,
    rows: u32,
    cell_m: f32,
    origin: [f32; 2],
) -> Result<[u32; 4], AcousticsError> {
    let r = &obb.rows;
    let finite = r.iter().all(|row| row.iter().all(|v| v.is_finite()));
    if !finite || !obb.half_width.is_finite() {
        return Err(AcousticsError::Static { index, why: "non-finite transform" });
    }
    let (cx, cz) = (r[0][3] as f64, r[2][3] as f64);
    let hx = 0.5 * (r[0][0].abs() + r[0][1].abs() + r[0][2].abs()) as f64 + BIN_MARGIN_M;
    let hz = 0.5 * (r[2][0].abs() + r[2][1].abs() + r[2][2].abs()) as f64 + BIN_MARGIN_M;
    let cell = cell_m as f64;
    let (ox, oz) = (origin[0] as f64, origin[1] as f64);
    let c0 = ((cx - hx - ox) / cell).floor();
    let c1 = ((cx + hx - ox) / cell).floor();
    let r0 = ((cz - hz - oz) / cell).floor();
    let r1 = ((cz + hz - oz) / cell).floor();
    if c1 < 0.0 || r1 < 0.0 || c0 >= cols as f64 || r0 >= rows as f64 {
        return Err(AcousticsError::Static { index, why: "outside the terrain grid" });
    }
    let clamp = |v: f64, n: u32| v.max(0.0).min((n - 1) as f64) as u32;
    let range = [clamp(c0, cols), clamp(r0, rows), clamp(c1, cols), clamp(r1, rows)];
    let area = (range[2] - range[0] + 1) as u64 * (range[3] - range[1] + 1) as u64;
    if area > MAX_CELLS_PER_STATIC as u64 {
        return Err(AcousticsError::Static { index, why: "covers more than 4,096 cells" });
    }
    Ok(range)
}

/// The scene's layout, validated. See [`super::GpuAcoustics::scene_bytes`].
pub(crate) fn layout(
    terrain: &TerrainGrid<'_>,
    statics: &[Obb],
    materials: u32,
    foliage: Option<&DensityGrid<'_>>,
) -> Result<SceneLayout, AcousticsError> {
    let (cols, rows) = (terrain.cols, terrain.rows);
    if cols > MAX_TERRAIN_SIDE || rows > MAX_TERRAIN_SIDE {
        return Err(AcousticsError::Limit {
            what: "terrain side",
            got: cols.max(rows) as u64,
            max: MAX_TERRAIN_SIDE as u64,
        });
    }
    let cells = cols as usize * rows as usize;
    if terrain.heights.len() != cells || terrain.material.len() != cells {
        return Err(AcousticsError::Shape("terrain slices are not cols x rows"));
    }
    if cells > 0 && !(terrain.cell_m > 0.0 && terrain.cell_m.is_finite()) {
        return Err(AcousticsError::Shape("cell size must be positive and finite"));
    }
    if !terrain.origin.iter().all(|v| v.is_finite()) {
        return Err(AcousticsError::Shape("terrain origin must be finite"));
    }
    if let Some(f) = foliage {
        if f.density.len() != cells {
            return Err(AcousticsError::Shape("foliage is not the terrain's shape"));
        }
    }
    if statics.len() > MAX_STATICS as usize {
        return Err(AcousticsError::Limit {
            what: "statics",
            got: statics.len() as u64,
            max: MAX_STATICS as u64,
        });
    }
    let mut list_len: u64 = 0;
    for (i, s) in statics.iter().enumerate() {
        if s.material >= materials {
            return Err(AcousticsError::Material { index: s.material, len: materials });
        }
        if cells == 0 {
            return Err(AcousticsError::Static { index: i as u32, why: "outside the terrain grid" });
        }
        let r = static_range(i as u32, s, cols, rows, terrain.cell_m, terrain.origin)?;
        list_len += (r[2] - r[0] + 1) as u64 * (r[3] - r[1] + 1) as u64;
    }
    let n = statics.len() as u32;
    let cells = cells as u32;
    let bytes4 = cells.div_ceil(4);
    let mut at = HEADER_WORDS;
    let mut take = |words: u32| {
        let start = at;
        at += words;
        (start, at)
    };
    let (off_heights, _) = take(cells);
    let (off_material, _) = take(bytes4);
    let (off_foliage, _) = take(if foliage.is_some() { bytes4 } else { 0 });
    let (off_statics, _) = take(16 * n);
    let (off_ranges, _) = take(4 * n);
    let (off_materials, staged_words) = take(4 * materials);
    let (off_inv, _) = take(12 * n);
    let (off_count, _) = take(cells);
    let (off_heads, _) = take(cells + 1);
    let (off_list, _) = take(list_len as u32);
    let (off_top, _) = take(cells);
    let mut pyr = [0u32; 3];
    let mut dims = [0u32; 6];
    for level in 0..PYRAMID_LEVELS as usize {
        let (c, r) = (cols.div_ceil(2 << level), rows.div_ceil(2 << level));
        dims[2 * level] = c;
        dims[2 * level + 1] = r;
        pyr[level] = take(c * r).0;
    }
    let end = take(0).1;
    // Every binding needs at least one element past the header, even for an empty scene.
    let total_words = end.max(HEADER_WORDS + 1);

    let mut h = [0u32; HEADER_WORDS as usize];
    h[H_COLS] = cols;
    h[H_ROWS] = rows;
    h[H_CELL] = terrain.cell_m.to_bits();
    h[H_ORIGIN_X] = terrain.origin[0].to_bits();
    h[H_ORIGIN_Z] = terrain.origin[1].to_bits();
    h[H_STATICS] = n;
    h[H_MATERIALS] = materials;
    h[H_HAS_FOLIAGE] = foliage.is_some() as u32;
    h[H_OFF_HEIGHTS] = off_heights;
    h[H_OFF_MATERIAL] = off_material;
    h[H_OFF_FOLIAGE] = off_foliage;
    h[H_OFF_STATICS] = off_statics;
    h[H_OFF_RANGES] = off_ranges;
    h[H_OFF_MATERIALS] = off_materials;
    h[H_OFF_INV] = off_inv;
    h[H_OFF_COUNT] = off_count;
    h[H_OFF_HEADS] = off_heads;
    h[H_OFF_LIST] = off_list;
    h[H_OFF_TOP] = off_top;
    h[H_OFF_PYR..H_OFF_PYR + 3].copy_from_slice(&pyr);
    h[H_PYR_DIMS..H_PYR_DIMS + 6].copy_from_slice(&dims);
    h[H_LIST_LEN] = list_len as u32;
    h[H_INV_CELL] = if cells > 0 { (1.0 / terrain.cell_m).to_bits() } else { 0 };
    h[H_RECT..H_RECT + 4].copy_from_slice(&[0, 0, cols, rows]);
    Ok(SceneLayout { header: h, staged_words, total_words })
}

/// Where packed bytes go: a CPU slice, or mapped staging memory lent as a
/// [`wgpu::WriteOnly`] (write-combined, so it is only ever written, in order).
pub type Out<'o> = wgpu::WriteOnly<'o, [u8]>;

fn put_bytes_as_words(out: &mut Out<'_>, word: u32, bytes: &[u8]) {
    let at = word as usize * 4;
    out.slice(at..at + bytes.len()).copy_from_slice(bytes);
    // The tail of the last word is zeroed so the staged bytes are fully determined.
    let end = at + bytes.len().div_ceil(4) * 4;
    out.slice(at + bytes.len()..end).fill(0);
}

/// Pack a scene into `out`. See [`super::GpuAcoustics::pack_scene`].
pub(crate) fn pack_scene(
    layout: &SceneLayout,
    terrain: &TerrainGrid<'_>,
    statics: &[Obb],
    materials: &[AcousticMaterial],
    foliage: Option<&DensityGrid<'_>>,
    mut out: Out<'_>,
) -> Result<(), AcousticsError> {
    let again = self::layout(terrain, statics, materials.len() as u32, foliage)?;
    if again != *layout {
        return Err(AcousticsError::Shape("the layout was made for a different scene"));
    }
    let need = layout.staged_bytes() as usize;
    if out.len() < need {
        return Err(AcousticsError::OutputTooSmall { need, got: out.len() });
    }
    for (b, &m) in terrain.material.iter().enumerate() {
        if m as usize >= materials.len() {
            let _ = b;
            return Err(AcousticsError::Material { index: m as u32, len: materials.len() as u32 });
        }
    }
    let out = &mut out;
    let h = &layout.header;
    put_bytes_as_words(out, 0, bytemuck::cast_slice(h));
    let heights: &[u8] = bytemuck::cast_slice(terrain.heights);
    put_bytes_as_words(out, h[H_OFF_HEIGHTS], heights);
    put_bytes_as_words(out, h[H_OFF_MATERIAL], terrain.material);
    if let Some(f) = foliage {
        put_bytes_as_words(out, h[H_OFF_FOLIAGE], f.density);
    }
    put_bytes_as_words(out, h[H_OFF_STATICS], bytemuck::cast_slice(statics));
    for (i, s) in statics.iter().enumerate() {
        let r = static_range(i as u32, s, terrain.cols, terrain.rows, terrain.cell_m, terrain.origin)?;
        put_bytes_as_words(out, h[H_OFF_RANGES] + 4 * i as u32, bytemuck::cast_slice(&r));
    }
    put_bytes_as_words(out, h[H_OFF_MATERIALS], bytemuck::cast_slice(materials));
    Ok(())
}

/// Bytes [`pack_terrain_rect`] writes for a rect: a 16-byte header and the heights.
///
/// # Arguments
///
/// * `rect` - the rectangle.
///
/// # Returns
///
/// `16 + 4 x cols x rows`.
///
/// # Examples
///
/// ```
/// use rs_physics::gpu::acoustics::{terrain_rect_bytes, GridRect};
///
/// assert_eq!(terrain_rect_bytes(GridRect { col: 0, row: 0, cols: 3, rows: 2 }), 40);
/// ```
pub fn terrain_rect_bytes(rect: GridRect) -> usize {
    16 + 4 * rect.cols as usize * rect.rows as usize
}

/// Pack a rect of new heights. See [`super::GpuAcoustics::pack_terrain_rect`].
pub(crate) fn pack_terrain_rect(
    rect: GridRect,
    heights: &[f32],
    mut out: Out<'_>,
) -> Result<usize, AcousticsError> {
    let n = rect.cols as usize * rect.rows as usize;
    if heights.len() != n || n == 0 {
        return Err(AcousticsError::Shape("rect heights are not cols x rows, or empty"));
    }
    if !heights.iter().all(|h| h.is_finite()) {
        return Err(AcousticsError::Shape("heights must be finite"));
    }
    let need = terrain_rect_bytes(rect);
    if out.len() < need {
        return Err(AcousticsError::OutputTooSmall { need, got: out.len() });
    }
    let head = [rect.col, rect.row, rect.cols, rect.rows];
    out.slice(..16).copy_from_slice(bytemuck::cast_slice(&head));
    out.slice(16..need).copy_from_slice(bytemuck::cast_slice(heights));
    Ok(need)
}

/// What [`super::GpuAcoustics::pack_dispatch`] packed, which
/// [`super::GpuAcoustics::encode`] needs to record it: the bytes and the counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchShape {
    /// Bytes packed, a multiple of 16.
    pub bytes: u64,
    /// Sources in the dispatch.
    pub sources: u32,
    /// Movers in the dispatch.
    pub movers: u32,
}

/// Bytes a dispatch of `sources` and `movers` packs to.
///
/// # Arguments
///
/// * `sources` - sources in the dispatch.
/// * `movers` - movers in the dispatch.
///
/// # Returns
///
/// `160 + 32 sources + 64 movers`.
///
/// # Examples
///
/// ```
/// use rs_physics::gpu::acoustics::dispatch_bytes;
///
/// // The design's typical set: 40 sources and 64 movers, about 5.5 KB up.
/// assert_eq!(dispatch_bytes(40, 64), 5_536);
/// ```
pub fn dispatch_bytes(sources: usize, movers: usize) -> usize {
    HEADER_BYTES as usize + 32 * sources + 64 * movers
}

/// Pack a dispatch. See [`super::GpuAcoustics::pack_dispatch`].
pub(crate) fn pack_dispatch(
    header: &DispatchHeader,
    sources: &[Source],
    movers: &[Obb],
    mut out: Out<'_>,
) -> Result<DispatchShape, AcousticsError> {
    if sources.len() > MAX_SOURCES as usize {
        return Err(AcousticsError::Limit {
            what: "sources",
            got: sources.len() as u64,
            max: MAX_SOURCES as u64,
        });
    }
    if movers.len() > MAX_MOVERS as usize {
        return Err(AcousticsError::Limit {
            what: "movers",
            got: movers.len() as u64,
            max: MAX_MOVERS as u64,
        });
    }
    let need = dispatch_bytes(sources.len(), movers.len());
    if out.len() < need {
        return Err(AcousticsError::OutputTooSmall { need, got: out.len() });
    }
    let mut h = *header;
    h.counts[0] = sources.len() as u32;
    h.counts[1] = movers.len() as u32;
    let head = HEADER_BYTES as usize;
    out.slice(..head).copy_from_slice(bytemuck::bytes_of(&h));
    let src_end = head + 32 * sources.len();
    out.slice(head..src_end).copy_from_slice(bytemuck::cast_slice(sources));
    out.slice(src_end..need).copy_from_slice(bytemuck::cast_slice(movers));
    Ok(DispatchShape {
        bytes: need as u64,
        sources: sources.len() as u32,
        movers: movers.len() as u32,
    })
}
