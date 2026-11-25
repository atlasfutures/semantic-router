//! Tests for the PyTorch pickle weights loading policy

use super::weights::*;
use candle_core::{DType, Device};
use std::path::Path;

#[test]
fn opt_in_values() {
    for v in ["1", "true", "TRUE", "True", " 1 "] {
        assert!(is_enabled(Some(v)), "{v:?} should enable pickle weights");
    }
    for v in ["", "0", "false", "yes", "on", "2"] {
        assert!(
            !is_enabled(Some(v)),
            "{v:?} should not enable pickle weights"
        );
    }
    assert!(!is_enabled(None), "unset must mean disabled");
}

#[test]
fn disabled_refuses_with_actionable_error() {
    let err = check_pth_allowed(Path::new("/models/x/pytorch_model.bin"), false)
        .expect_err("pickle weights must be refused when not opted in");
    let msg = err.to_string();
    assert!(msg.contains("/models/x/pytorch_model.bin"), "{msg}");
    assert!(msg.contains(ALLOW_PTH_WEIGHTS_ENV), "{msg}");
    assert!(msg.contains("model.safetensors"), "{msg}");
}

#[test]
fn enabled_allows() {
    check_pth_allowed(Path::new("/models/x/pytorch_model.bin"), true)
        .expect("pickle weights must be allowed when opted in");
}

#[test]
fn loader_refuses_before_opening_file_when_disabled() {
    if pth_weights_allowed() {
        // The suite runs with the opt-in set; the disabled path is covered above.
        return;
    }
    // A path that does not exist: the error must be the policy refusal,
    // not a file or parse error, so the file is never opened.
    let err = var_builder_from_pth("/nonexistent/pytorch_model.bin", DType::F32, &Device::Cpu)
        .err()
        .expect("must refuse when not opted in");
    assert!(err.to_string().contains(ALLOW_PTH_WEIGHTS_ENV), "{err}");
}

#[test]
fn loader_reaches_candle_when_enabled() {
    if !pth_weights_allowed() {
        return;
    }
    // With the opt-in set, the call gets past the policy check to candle,
    // which fails on the missing file with its own error.
    let err = var_builder_from_pth("/nonexistent/pytorch_model.bin", DType::F32, &Device::Cpu)
        .err()
        .expect("missing file must fail in candle");
    assert!(!err.to_string().contains(ALLOW_PTH_WEIGHTS_ENV), "{err}");
}
