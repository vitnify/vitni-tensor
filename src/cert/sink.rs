//! Cert sink — runtime wiring without a the host runtime dependency.
//!
//! vitni-tensor stays standalone: it never depends on the host runtime directly.
//! Instead it defines this trait, and the user (the host application
//! binary, or downstream wrapper crate) plugs in an impl that calls
//! the real `the runtime's cert API` syscalls.
//!
//! Cross-verification property: the software cert digest (computed
//! by `CertBuilder::finalize`) MUST equal the runtime-issued cert
//! digest (computed inside the kernel from the same `declare_input`/
//! `declare_output` calls). Because the binding format is identical
//! — the runtime's runtime cert system uses the same canonical hash over
//! length-prefixed (name, bytes) pairs — both paths produce the same
//! 32-byte digest. The runtime just adds its kernel signature on
//! top so verifiers can trust the cert was issued by an authentic
//! the host runtime process.
//!
//! # runtime-side wiring
//!
//! In a the host application binary using vitni-tensor:
//!
//! ```ignore
//! struct RuntimeSink;
//! impl vitni_tensor::cert::CertSink for RuntimeSink {
//!     type Error = host::Error;
//!     fn on_request(&mut self, tier: u8) -> Result<(), Self::Error> {
//!         the runtime's cert API(tier).map(|_| ())
//!     }
//!     fn on_input(&mut self, name: &str, bytes: &[u8]) -> Result<(), Self::Error> {
//!         the runtime's cert API(name.as_bytes(), bytes).map(|_| ())
//!     }
//!     fn on_output(&mut self, name: &str, bytes: &[u8]) -> Result<(), Self::Error> {
//!         the runtime's cert API(name.as_bytes(), bytes).map(|_| ())
//!     }
//!     fn on_finalize(&mut self) -> Result<u64, Self::Error> {
//!         the runtime's cert API()
//!     }
//! }
//! ```
//!
//! Then call `CertBuilder::finalize_with_sink(&mut RuntimeSink, tier)`.
//! Both the kernel-issued cert and the returned software `Cert` will
//! carry the same 32-byte digest.

use alloc::vec::Vec;

/// Runtime-side cert pipeline. The default `NullSink` does nothing
/// (software-only path); a runtime-side impl forwards to `the runtime's cert API`.
pub trait CertSink {
    /// Per-sink error type. Use `core::convert::Infallible` for sinks
    /// that can never fail (the null + recording sinks both do).
    type Error;

    /// Called once at the start of `finalize_with_sink` with the cert
    /// tier (matches `the runtime's cert syscall`'s tier argument).
    fn on_request(&mut self, tier: u8) -> Result<(), Self::Error>;

    /// Called for each declared input, in declaration order.
    fn on_input(&mut self, name: &str, bytes: &[u8]) -> Result<(), Self::Error>;

    /// Called for each declared per-op record (PerOp mode), in
    /// declaration order, AFTER inputs and BEFORE outputs.
    ///
    /// Default impl is a no-op so existing sinks (including the
    /// downstream `RuntimeSink`) compile without modification.
    /// Sinks that want runtime-side per-op attestation override
    /// this to forward each record to `the runtime's cert API`
    /// (kernel the runtime's cert syscall, Phase 4a kernel-side work).
    fn on_op(&mut self, _op: &super::builder::OpRecord) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Called for each declared activation record (Phase 4b), in
    /// declaration order, AFTER ops and BEFORE outputs.
    ///
    /// Default impl is a no-op so existing sinks (NullSink,
    /// pre-Phase-4b RuntimeSink implementations) compile without
    /// modification. Sinks that want runtime-side activation
    /// attestation override this to forward each record to
    /// `the runtime's cert API`
    /// (kernel the runtime's cert syscall, Phase 4b kernel work).
    fn on_activation(
        &mut self,
        _act: &super::builder::ActivationRecord,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Called for each declared causal-intervention record (Phase 4c),
    /// in declaration order, AFTER activations and BEFORE outputs.
    /// Default no-op for sink-compat. Runtime sinks forward to
    /// `the runtime's cert API`
    /// (kernel the runtime's cert syscall).
    fn on_intervention(
        &mut self,
        _intv: &super::builder::InterventionRecord,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Called for each declared output, in declaration order.
    fn on_output(&mut self, name: &str, bytes: &[u8]) -> Result<(), Self::Error>;

    /// Called once at the end. Returns the runtime-issued cert ID
    /// (for the host runtime, the `u64` from the runtime finalize call). Sinks that
    /// don't issue an ID may return 0.
    fn on_finalize(&mut self) -> Result<u64, Self::Error>;
}

/// No-op sink. Used when the caller doesn't want runtime submission
/// (host tests, dry-run mode). All methods succeed without side effects.
pub struct NullSink;

impl CertSink for NullSink {
    type Error = core::convert::Infallible;
    fn on_request(&mut self, _tier: u8) -> Result<(), Self::Error> {
        Ok(())
    }
    fn on_input(&mut self, _name: &str, _bytes: &[u8]) -> Result<(), Self::Error> {
        Ok(())
    }
    fn on_output(&mut self, _name: &str, _bytes: &[u8]) -> Result<(), Self::Error> {
        Ok(())
    }
    fn on_finalize(&mut self) -> Result<u64, Self::Error> {
        Ok(0)
    }
}

/// One event in a recorded cert pipeline. Used by `RecordingSink` for
/// test introspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkEvent {
    /// `on_request(tier)` called.
    Request {
        /// Cert tier.
        tier: u8,
    },
    /// `on_input(name, bytes)` called.
    Input {
        /// Field name.
        name: alloc::string::String,
        /// Field bytes.
        bytes: Vec<u8>,
    },
    /// `on_op(record)` called.
    Op {
        /// Op record passed in.
        record: super::builder::OpRecord,
    },
    /// `on_activation(record)` called.
    Activation {
        /// Activation record passed in.
        record: super::builder::ActivationRecord,
    },
    /// `on_intervention(record)` called.
    Intervention {
        /// Intervention record passed in.
        record: super::builder::InterventionRecord,
    },
    /// `on_output(name, bytes)` called.
    Output {
        /// Field name.
        name: alloc::string::String,
        /// Field bytes.
        bytes: Vec<u8>,
    },
    /// `on_finalize()` called.
    Finalize,
}

/// Sink that records every call into a `Vec<SinkEvent>` for test
/// inspection. Used to verify runtime-side declarations match the
/// software-side digest field-for-field.
#[derive(Debug, Default)]
pub struct RecordingSink {
    /// All events recorded, in call order.
    pub events: Vec<SinkEvent>,
    /// Cert ID to return from `on_finalize`. Defaults to 0; tests can
    /// set to anything to verify the caller surfaces it correctly.
    pub finalize_id: u64,
}

impl CertSink for RecordingSink {
    type Error = core::convert::Infallible;
    fn on_request(&mut self, tier: u8) -> Result<(), Self::Error> {
        self.events.push(SinkEvent::Request { tier });
        Ok(())
    }
    fn on_input(&mut self, name: &str, bytes: &[u8]) -> Result<(), Self::Error> {
        self.events.push(SinkEvent::Input {
            name: alloc::string::String::from(name),
            bytes: bytes.to_vec(),
        });
        Ok(())
    }
    fn on_op(&mut self, op: &super::builder::OpRecord) -> Result<(), Self::Error> {
        self.events.push(SinkEvent::Op { record: op.clone() });
        Ok(())
    }
    fn on_activation(
        &mut self,
        act: &super::builder::ActivationRecord,
    ) -> Result<(), Self::Error> {
        self.events.push(SinkEvent::Activation { record: act.clone() });
        Ok(())
    }
    fn on_intervention(
        &mut self,
        intv: &super::builder::InterventionRecord,
    ) -> Result<(), Self::Error> {
        self.events.push(SinkEvent::Intervention { record: intv.clone() });
        Ok(())
    }
    fn on_output(&mut self, name: &str, bytes: &[u8]) -> Result<(), Self::Error> {
        self.events.push(SinkEvent::Output {
            name: alloc::string::String::from(name),
            bytes: bytes.to_vec(),
        });
        Ok(())
    }
    fn on_finalize(&mut self) -> Result<u64, Self::Error> {
        self.events.push(SinkEvent::Finalize);
        Ok(self.finalize_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_sink_no_ops() {
        let mut s = NullSink;
        assert!(s.on_request(0).is_ok());
        assert!(s.on_input("k", b"v").is_ok());
        assert!(s.on_output("k", b"v").is_ok());
        assert_eq!(s.on_finalize().unwrap(), 0);
    }

    #[test]
    fn recording_sink_captures_events_in_order() {
        let mut s = RecordingSink::default();
        s.finalize_id = 99;
        s.on_request(2).unwrap();
        s.on_input("a", b"1").unwrap();
        s.on_input("b", b"22").unwrap();
        s.on_output("out", b"xyz").unwrap();
        let id = s.on_finalize().unwrap();
        assert_eq!(id, 99);
        assert_eq!(s.events.len(), 5);
        match &s.events[0] {
            SinkEvent::Request { tier } => assert_eq!(*tier, 2),
            _ => panic!("expected Request"),
        }
        match &s.events[1] {
            SinkEvent::Input { name, bytes } => {
                assert_eq!(name, "a");
                assert_eq!(bytes, b"1");
            }
            _ => panic!("expected Input"),
        }
        assert_eq!(s.events[4], SinkEvent::Finalize);
    }
}
