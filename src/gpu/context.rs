//! GPU context management for wgpu compute operations

use wgpu::{Device, Queue, Instance, Adapter};

/// GPU context holding the wgpu device and queue
///
/// This is the main entry point for GPU operations. Create one context
/// and share it across multiple simulations.
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    #[allow(dead_code)]
    adapter: Adapter,
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
                required_features: wgpu::Features::empty(),
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
            adapter,
        })
    }

    /// Get information about the GPU adapter
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}
