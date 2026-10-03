//! GPU context management for wgpu compute operations

use wgpu::{Adapter, Device, Features, Instance, Queue};

/// GPU context holding the wgpu device and queue
///
/// This is the main entry point for GPU operations. Create one context
/// and share it across multiple simulations.
///
/// A context either opens its own adapter and device ([`GpuContext::new`],
/// [`GpuContext::with_features`]) or wraps a device an engine already owns
/// ([`GpuContext::from_device`]), so rs_physics' compute passes run on the same device
/// as the engine's draws and their buffers can be bound by both.
pub struct GpuContext {
    /// The device every buffer, texture and pipeline of this context is created on.
    pub device: Device,
    /// The queue writes and submissions go through.
    pub queue: Queue,
    /// The adapter, when this context opened the device itself; `None` for a device
    /// handed in by [`GpuContext::from_device`].
    adapter: Option<Adapter>,
}

impl GpuContext {
    /// Create a new GPU context
    ///
    /// This will request a GPU adapter and create a device with compute capabilities.
    /// Returns None if no suitable GPU is available.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let gpu = GpuContext::new().expect("No GPU available");
    /// ```
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Async version of context creation
    pub async fn new_async() -> Option<Self> {
        Self::open(Features::empty()).await
    }

    /// Open a context of its own whose device has every feature of `wanted` that the
    /// adapter offers, and no other.
    ///
    /// For callers that can use a feature when it is there and do without when it is
    /// not: timestamp queries for a benchmark, `FLOAT32_FILTERABLE` for a hardware-filtered
    /// `f32` field. Read [`Device::features`] afterwards to learn what was granted.
    ///
    /// # Arguments
    ///
    /// * `wanted` - the features to ask for where the adapter has them.
    ///
    /// # Returns
    ///
    /// The context, or `None` when no adapter or device could be opened.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::GpuContext;
    /// if let Some(gpu) = GpuContext::with_features(wgpu::Features::TIMESTAMP_QUERY) {
    ///     let timed = gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    ///     println!("timestamps: {timed}");
    /// }
    /// ```
    pub fn with_features(wanted: Features) -> Option<Self> {
        pollster::block_on(Self::open(wanted))
    }

    async fn open(wanted: Features) -> Option<Self> {
        // wgpu 30: no display handle, since this context never presents; `_from_env` so
        // `WGPU_BACKEND` is honoured the way the engine's context honours it.
        let instance = Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("rs_physics GPU"),
                required_features: wanted & adapter.features(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .ok()?;

        log::info!("GPU initialized: {:?}", adapter.get_info().name);

        Some(Self {
            device,
            queue,
            adapter: Some(adapter),
        })
    }

    /// Wrap a device and queue the caller already owns, so rs_physics' passes run on
    /// the engine's device and the buffers they write can be drawn from directly.
    ///
    /// **Owned handles, not borrows.** wgpu 30's [`Device`] and [`Queue`] are reference
    /// counted and `Clone`; a clone is a second handle to the same device, not a second
    /// device. Taking them by value keeps this context free of a lifetime tied to the
    /// engine's own struct, so a pool built on it can live in whatever the engine stores
    /// it in, and the engine keeps its handles: pass `device.clone()` and `queue.clone()`.
    ///
    /// The context can only use the features and limits the device was opened with;
    /// anything that needs more says so when it is built.
    ///
    /// # Arguments
    ///
    /// * `device` - the engine's device.
    /// * `queue` - that device's queue.
    ///
    /// # Returns
    ///
    /// The context. [`Self::adapter_info`] still answers, from the device.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rs_physics::gpu::GpuContext;
    /// let own = GpuContext::new().expect("a GPU");
    /// // An engine hands over clones of its handles and keeps its own.
    /// let shared = GpuContext::from_device(own.device.clone(), own.queue.clone());
    /// assert_eq!(shared.adapter_info().name, own.adapter_info().name);
    /// ```
    pub fn from_device(device: Device, queue: Queue) -> Self {
        Self {
            device,
            queue,
            adapter: None,
        }
    }

    /// Get information about the GPU adapter
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        match &self.adapter {
            Some(adapter) => adapter.get_info(),
            None => self.device.adapter_info(),
        }
    }
}
