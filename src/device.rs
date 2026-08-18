//! Where a tensor lives. `Cpu` is always available; `Gpu(_)` means
//! storage is a `an opaque GPU backend handle` handle and ops dispatch
//! via `SYS_GPU_*`.

/// Device identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU memory (Vec-backed storage in process address space).
    Cpu,
    /// Kernel-managed GPU memory. `u8` is the device index per
    /// `SYS_GPU_INFO`. Index 0 == "first available GPU."
    Gpu(u8),
}

impl Device {
    /// Short tag for error messages.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            // We can't format the GPU index in no_std without alloc
            // tricks; the tag elides it. For diagnostics we'd add a
            // `pub fn description(&self) -> alloc::string::String`.
            Device::Gpu(_) => "gpu",
        }
    }

    /// `true` if both tensors live on a device where ops can run
    /// without a CPU↔GPU bounce.
    pub const fn same(self, other: Self) -> bool {
        match (self, other) {
            (Device::Cpu, Device::Cpu) => true,
            (Device::Gpu(a), Device::Gpu(b)) => a == b,
            _ => false,
        }
    }
}
