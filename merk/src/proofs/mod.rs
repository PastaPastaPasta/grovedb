//! Merk proofs

#[cfg(any(feature = "minimal", feature = "verify"))]
pub mod branch;
#[cfg(feature = "minimal")]
pub mod chunk;
/// Query proof generation and verification.
pub mod query;
#[cfg(any(feature = "minimal", feature = "verify"))]
pub mod tree;

// Re-export Op and Node from grovedb-query
// Re-export hex_to_ascii for use in verify.rs
pub use grovedb_query::hex_to_ascii;
// Re-export encode_into from grovedb-query
#[cfg(feature = "minimal")]
pub use grovedb_query::proofs::encode_into;
#[cfg(any(feature = "minimal", feature = "verify"))]
pub use grovedb_query::proofs::{Node, Op};
#[cfg(any(feature = "minimal", feature = "verify"))]
pub use query::Query;
#[cfg(any(feature = "minimal", feature = "verify"))]
pub use tree::{execute, Tree};

/// Decoder iterates over proof bytes, yielding Op values with merk Error type.
///
/// This wraps grovedb-query's Decoder, converting its error type to merk's
/// Error. Like the wrapped decoder, it yields a decoding error once and then
/// stays exhausted; the invalid bytes remain counted by
/// [`Self::remaining_bytes`].
#[cfg(any(feature = "minimal", feature = "verify"))]
pub struct Decoder<'a> {
    inner: grovedb_query::proofs::Decoder<'a>,
}

#[cfg(any(feature = "minimal", feature = "verify"))]
impl<'a> Decoder<'a> {
    /// Create a new Decoder from proof bytes
    pub const fn new(proof_bytes: &'a [u8]) -> Self {
        Decoder {
            inner: grovedb_query::proofs::Decoder::new(proof_bytes),
        }
    }

    /// Returns the number of bytes not yet consumed by the decoder.
    pub const fn remaining_bytes(&self) -> usize {
        self.inner.remaining_bytes()
    }
}

#[cfg(any(feature = "minimal", feature = "verify"))]
impl Iterator for Decoder<'_> {
    type Item = Result<Op, crate::error::Error>;

    // `match` rather than `Option::map`, to keep closures off the verifier
    // path.
    #[allow(clippy::manual_map)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next_core() {
            None => None,
            Some(item) => Some(merk_decode_item(item)),
        }
    }
}

/// One decoded item with its error mapped to `Error::InvalidProofError`
/// (the message `op_decode_error_message` gives). A function of its own so
/// `next` moves the whole item out of `next_core`'s result.
#[cfg(any(feature = "minimal", feature = "verify"))]
fn merk_decode_item(
    item: Result<Op, grovedb_query::proofs::OpDecodeError>,
) -> Result<Op, crate::error::Error> {
    match item {
        Ok(op) => Ok(op),
        Err(e) => Err(crate::error::Error::InvalidProofError(
            grovedb_query::proofs::op_decode_error_message(&e),
        )),
    }
}

// The wrapped decoder is fused (one error, then `None` forever), and `next`
// only maps its items, so the wrapper upholds the same contract.
#[cfg(any(feature = "minimal", feature = "verify"))]
impl std::iter::FusedIterator for Decoder<'_> {}

/// Re-export encoding module for backward compatibility
#[cfg(any(feature = "minimal", feature = "verify"))]
pub mod encoding {
    pub use grovedb_query::proofs::encode_into;

    pub use super::Decoder;
}

#[cfg(all(test, any(feature = "minimal", feature = "verify")))]
mod tests {
    use super::{Decoder, Op};

    #[test]
    fn decoder_yields_one_error_then_stays_exhausted() {
        // A valid Parent op, then an unknown opcode followed by one more byte.
        let bytes = [0x10, 0xFF, 0x10];
        let mut decoder = Decoder::new(&bytes);
        assert!(matches!(decoder.next(), Some(Ok(Op::Parent))));
        assert_eq!(decoder.remaining_bytes(), 2);
        assert!(matches!(decoder.next(), Some(Err(_))));
        for _ in 0..3 {
            assert!(decoder.next().is_none());
            assert_eq!(decoder.remaining_bytes(), 2);
        }
    }

    #[test]
    fn decoder_error_is_invalid_proof_error_with_old_message() {
        for (bytes, message) in [
            (
                vec![0xFFu8],
                "failed to decode an proofs::Op structure (UnexpectedByte: 255)",
            ),
            (
                vec![0x01u8, 0x00],
                "failed to decode an proofs::Op structure (failed to fill whole buffer)",
            ),
        ] {
            let mut decoder = Decoder::new(&bytes);
            match decoder.next() {
                Some(Err(crate::error::Error::InvalidProofError(s))) => assert_eq!(s, message),
                other => panic!("unexpected {other:?}"),
            }
            assert!(decoder.next().is_none());
        }
    }
}
