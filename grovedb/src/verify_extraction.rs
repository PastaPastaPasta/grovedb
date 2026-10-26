//! Extraction roots for the formal-verification toolchain.
//!
//! Compiled only with the `verify-extraction` feature, which no build of
//! GroveDB enables: the toolchain (Charon + Aeneas) translates this crate's
//! own functions and everything they call, so these one-line forwards make
//! Merk's public query-proof entry points part of what it translates. Each
//! returns exactly what the Merk function returns; the theorems are stated
//! about the Merk functions themselves.

use grovedb_costs::CostResult;
use grovedb_merk::{
    error::Error as MerkError,
    proofs::{
        query::{
            boundaries_in_proof, key_exists_as_boundary_in_proof, ProofVerificationResult,
            QueryProofVerify,
        },
        tree::{execute, Tree},
        Node, Op, Query,
    },
    CryptoHash,
};

/// `QueryProofVerify::verify_proof` for `Query`.
pub fn merk_verify_proof(
    q: &Query,
    bytes: &[u8],
    limit: Option<u16>,
    left_to_right: bool,
    expected_hash: CryptoHash,
) -> CostResult<ProofVerificationResult, MerkError> {
    q.verify_proof(bytes, limit, left_to_right, expected_hash)
}

/// `grovedb_merk::proofs::query::boundaries_in_proof`.
pub fn merk_boundaries_in_proof(proof_bytes: &[u8]) -> Result<Vec<Vec<u8>>, MerkError> {
    boundaries_in_proof(proof_bytes)
}

/// `grovedb_merk::proofs::query::key_exists_as_boundary_in_proof`.
pub fn merk_key_exists_as_boundary_in_proof(
    proof_bytes: &[u8],
    key: &[u8],
) -> Result<bool, MerkError> {
    key_exists_as_boundary_in_proof(proof_bytes, key)
}

/// `grovedb_merk::proofs::tree::execute`, as generic as the original.
pub fn merk_execute<I, F>(ops: I, collapse: bool, visit_node: F) -> CostResult<Tree, MerkError>
where
    I: IntoIterator<Item = Result<Op, MerkError>>,
    F: FnMut(&Node) -> Result<(), MerkError>,
{
    execute(ops, collapse, visit_node)
}
