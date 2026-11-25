//! Loading policy for PyTorch pickle (`.pth` / `pytorch_model.bin`) weights.
//!
//! candle reads pickle checkpoints through a zip archive parser. Every model the
//! router ships has `model.safetensors`, so pickle loading is off by default and
//! must be enabled explicitly with `SR_ALLOW_PTH_WEIGHTS=1` (or `true`).
//! When it is off, a model that only has pickle weights fails to load with an
//! error that names the file and the opt-in, instead of being skipped.

use anyhow::{bail, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::Path;

/// Environment variable that enables loading PyTorch pickle weights.
pub const ALLOW_PTH_WEIGHTS_ENV: &str = "SR_ALLOW_PTH_WEIGHTS";

/// Whether an `SR_ALLOW_PTH_WEIGHTS` value enables pickle loading ("1" or "true").
pub(crate) fn is_enabled(value: Option<&str>) -> bool {
    matches!(value.map(str::trim), Some(v) if v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Whether PyTorch pickle weights may be loaded in this process.
pub fn pth_weights_allowed() -> bool {
    is_enabled(std::env::var(ALLOW_PTH_WEIGHTS_ENV).ok().as_deref())
}

pub(crate) fn check_pth_allowed(path: &Path, allowed: bool) -> Result<()> {
    if !allowed {
        bail!(
            "refusing to load PyTorch pickle weights {}: provide model.safetensors \
             (or a .safetensors adapter), or set {}=1 to allow .pth/.bin weights",
            path.display(),
            ALLOW_PTH_WEIGHTS_ENV
        );
    }
    Ok(())
}

/// `VarBuilder::from_pth`, gated on `SR_ALLOW_PTH_WEIGHTS`.
pub fn var_builder_from_pth<P: AsRef<Path>>(
    path: P,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    check_pth_allowed(path.as_ref(), pth_weights_allowed())?;
    Ok(VarBuilder::from_pth(path, dtype, device)?)
}
