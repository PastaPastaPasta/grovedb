use std::collections::{BTreeMap, BTreeSet};

use grovedb_merk::{
    calculate_chunk_depths, calculate_max_tree_depth_from_count,
    element::tree_type::ElementTreeTypeExtensions,
    proofs::{
        query::{
            PathKey, ProvedKeyOptionalValue, QueryProofVerify, SumBudgetRead, VerifyOptions,
            PROOF_VERSION_LATEST,
        },
        Decoder, Node, Op, Query,
    },
    tree::{axes_digest, combine_hash, combine_hash_three, value_hash, NULL_HASH},
    CryptoHash, TreeFeatureType,
};
use grovedb_version::{
    check_grovedb_v0, version::GroveVersion, TryFromVersioned, TryIntoVersioned,
};

use super::position_intervals::PositionIntervals;
#[cfg(feature = "proof_debug")]
use crate::operations::proof::util::{
    hex_to_ascii, path_as_slices_hex_to_ascii, path_hex_to_ascii,
};
use crate::{
    operations::proof::{
        indexed_axis::AxisEntries,
        util::{ProvedPathKeyOptionalValue, ProvedPathKeyValue, ProvedPathKeyValues},
        AxisDescentProof, GroveDBProof, GroveDBProofV0, GroveDBProofV1, LayerProof,
        MerkOnlyLayerProof, ProofBytes, ProveOptions,
    },
    query::{GroveTrunkQueryResult, PathTrunkChunkQuery, SinglePathSubquery},
    query_result_type::PathKeyOptionalElementTrio,
    Element, Error, GroveDb, PathQuery,
};

/// One axis-ordered layer verified during a V1 walk: the full path to
/// the indexed tree whose axis was read, plus the traversal's verified
/// answer. Collected into a side channel because axis entries are
/// `(aggregate_value, original_key)` pairs, not elements — they do not
/// belong in the trio result set.
#[derive(Debug)]
pub(crate) struct AxisWalkOutcome {
    /// Full path (from the GroveDB root) of the indexed tree.
    pub(crate) path: Vec<Vec<u8>>,
    /// The traversal's verified answer.
    pub(crate) result: AxisWalkResult,
}

/// The verified answer of one read-mode layer, by shape family.
#[derive(Debug)]
pub(crate) enum AxisWalkResult {
    /// `TopK` / `Bounded`: the entries in walk order. `skipped` is the
    /// count-commitment-attested skip for paginated traversals, `None`
    /// for bounded ones (which do not skip).
    Entries {
        entries: AxisEntries,
        skipped: Option<u64>,
    },
    /// `RankOfKey`: the attested 0-based rank of the queried key.
    Rank { rank: u64 },
    /// `AggregateOverValueRange`: the attested aggregate over the value range.
    Aggregate { value: i128 },
    /// Sum-budget window: the matched `(key, value)` pairs, their net
    /// total, and the replay-attested stop condition.
    SumBudget {
        matches: Vec<(Vec<u8>, i64)>,
        total: i64,
        stop: SumBudgetStop,
    },
}

/// Why a verified sum-budget walk stopped — attested by the verifier's
/// fold replay over the proved window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumBudgetStop {
    /// The running net sum reached the budget.
    BudgetReached,
    /// The match cap fired first.
    MatchLimitReached,
    /// The grove-version global scan cap fired first; more elements may
    /// exist beyond the window.
    HardScanCapReached,
    /// The query's ranges were exhausted before any stop condition.
    Exhausted,
}

/// The parent Merk row a lower layer hangs off: the element bytes the proof
/// carries, and the value hash the parent Merk commits for them.
///
/// A lower layer is bound to its parent when
/// `combine_hash(H(value_bytes), lower_root) == value_hash`. The append-only
/// (non-Merk) lower layers check this as soon as they have computed their
/// root, before they report any row, as the standalone
/// `BulkAppendTreeProof::verify_against_query` does. The MMR and dense
/// layers decode their leaves and entries while computing that root, which
/// the proof bytes bound; the bulk and commitment-tree layers decode chunk
/// blobs only after binding. This is defense in depth, not a resource
/// bound. The parent row is itself unauthenticated until the caller compares
/// the returned root with a trusted one, so a forger can make it consistent
/// with any lower root. Each extraction must bound its own work.
#[derive(Clone, Copy)]
struct ParentCommitment<'a> {
    value_bytes: &'a [u8],
    value_hash: &'a CryptoHash,
}

impl ParentCommitment<'_> {
    /// `Ok` when the parent commits to `lower_hash`; otherwise the value hash
    /// it would have to commit, `combine_hash(H(value_bytes), lower_hash)`.
    fn check(&self, lower_hash: &CryptoHash) -> Result<(), CryptoHash> {
        let combined_root_hash = combine_hash(value_hash(self.value_bytes).value(), lower_hash)
            .value()
            .to_owned();
        if self.value_hash != &combined_root_hash {
            return Err(combined_root_hash);
        }
        Ok(())
    }

    fn bind(&self, query: &PathQuery, lower_hash: &CryptoHash) -> Result<(), Error> {
        self.check(lower_hash).map_err(|combined_root_hash| {
            invalid_proof(
                query,
                format!(
                    "V1 mismatch in lower layer hash, expected {}, got {}",
                    hex::encode(self.value_hash),
                    hex::encode(combined_root_hash)
                ),
            )
        })
    }
}

/// The root an MMR lower layer proof yields, before it is bound to the
/// parent row.
enum MmrLowerRoot {
    /// An empty MMR (`mmr_size == 0`, no leaves); its root is `[0u8; 32]`.
    Empty,
    /// The root computed from the proof, with the leaves it verified and the
    /// element's `mmr_size` (equal to the proof's).
    Proved {
        root: CryptoHash,
        leaves: grovedb_merkle_mountain_range::VerifiedLeaves,
        mmr_size: u64,
    },
}

/// The root a dense lower layer proof yields, before it is bound to the
/// parent row.
enum DenseLowerRoot {
    /// Root only (`report_contents` unset): nothing is reported.
    RootOnly(CryptoHash),
    /// The root `verify_for_query` computed, with the entries the query
    /// selects and the query's direction.
    Selected {
        root: CryptoHash,
        entries: Vec<(u16, Vec<u8>)>,
        left_to_right: bool,
    },
}

/// `Error::InvalidProof` for a V1 verifier rejection. Out of line so the
/// verifier clones the query only on the error path.
#[cold]
fn invalid_proof(path_query: &PathQuery, message: String) -> Error {
    Error::InvalidProof(path_query.clone(), message)
}

/// `Error::CorruptedPath` for a V1 layer whose `path` is not part of
/// `query`; `context` prefixes the message. Out of line so the message is
/// formatted only on the error path.
#[cold]
fn corrupted_layer_path(context: &str, path: &[&[u8]], query: &PathQuery) -> Error {
    Error::CorruptedPath(format!(
        "{}: path {} should be part of path_query {}",
        context,
        path.iter().map(hex::encode).collect::<Vec<_>>().join("/"),
        query
    ))
}

/// Owned copy of a borrowed path, component by component, in order, rather
/// than `iter().map(..).collect()`.
fn path_to_owned(path: &[&[u8]]) -> Vec<Vec<u8>> {
    let mut owned = Vec::with_capacity(path.len());
    push_owned_components(path, 0, &mut owned);
    owned
}

/// Push `path[i..]` onto `owned`, in order. Recursive rather than a loop; the
/// depth is `path.len() - i`, and the paths passed here are a V1 layer path
/// (at most one component per proof level, `MAX_PROOF_DEPTH`) plus at most
/// two keys.
fn push_owned_components(path: &[&[u8]], i: usize, owned: &mut Vec<Vec<u8>>) {
    if i < path.len() {
        owned.push(path[i].to_vec());
        push_owned_components(path, i + 1, owned);
    }
}

/// Whether a V1 row whose lower layer the proof does not carry is reported
/// as a result row (`verify_terminal_row`): an item, or an element no subquery
/// descends into that the options allow to be reported (an empty `Tree` only
/// with `include_empty_trees_in_result`). Evaluated left to right as
/// `item || !descends && (include_empty || !empty_tree)`.
fn is_terminal_row(
    element: &Element,
    internal_query: &SinglePathSubquery<'_>,
    key: &[u8],
    options: &VerifyOptions,
) -> bool {
    if element.is_any_item() {
        return true;
    }
    if internal_query.has_subquery_or_matching_in_path_on_key(key) {
        return false;
    }
    if options.include_empty_trees_in_result {
        return true;
    }
    !is_empty_plain_tree(element)
}

/// `matches!(element, Element::Tree(None, _))`: a plain `Tree` with no root
/// key.
fn is_empty_plain_tree(element: &Element) -> bool {
    matches!(element, Element::Tree(None, _))
}

/// Whether a layer is a level `query_items_at_path` synthesized for a path
/// component and its items are exactly one `QueryItem::Key`, i.e.
/// `synthesized_path_component && matches!(items.as_slice(), [Key(_)])`.
fn is_single_key_synthesized_level(internal_query: &SinglePathSubquery<'_>) -> bool {
    if !internal_query.synthesized_path_component {
        return false;
    }
    let items = internal_query.items.as_slice();
    if items.len() != 1 {
        return false;
    }
    matches!(items[0], grovedb_merk::proofs::query::QueryItem::Key(_))
}

/// Whether a terminally reported tree must come with a verified child hash:
/// every non-empty Merk tree, and from `terminal_non_merk_tree_child_hash`
/// version 1 on, every tree `is_non_empty_tree` covers.
fn terminal_requires_child_hash(element: &Element, grove_version: &GroveVersion) -> bool {
    element.is_non_empty_merk_tree()
        || (grove_version
            .grovedb_versions
            .operations
            .proof
            .terminal_non_merk_tree_child_hash
            >= 1
            && element.is_non_empty_tree())
}

/// The limit a V1 layer's Merk proof runs under: the tighter of the global
/// and instance budgets for a pure terminal layer (`HasSubquery::NoSubquery`),
/// the global budget otherwise. A function of its own so the choice reads the
/// limit state through one shared borrow, matching every variant (no
/// wildcard arm).
fn v1_layer_limit(
    has_subquery: &crate::query::HasSubquery<'_>,
    limit_state: &super::V1LimitState,
    frame_instance: Option<u16>,
) -> Option<u16> {
    match has_subquery {
        crate::query::HasSubquery::NoSubquery => limit_state.effective_layer_limit(frame_instance),
        crate::query::HasSubquery::Always | crate::query::HasSubquery::Conditionally(_) => {
            limit_state.global
        }
    }
}

/// Charge an empty V1 layer to the global budget when the prove options ask
/// for it (`decrease_limit_on_empty_sub_query_result`); per-instance budgets
/// bound rows, so they are not charged.
fn charge_empty_v1_layer(limit_state: &mut super::V1LimitState, decrease_limit: bool) {
    if decrease_limit {
        limit_state.charge_empty_layer();
    }
}

/// The `mmr_size` of an `MmrTree` element, or the error for an MMR proof
/// attached to any other element. The element-kind checks of the non-Merk
/// lower layers (`mmr_size_of`, `bulk_dims_of`, `dense_dims_of`) run before
/// their proof decoding, in functions of their own, so the kind is decided
/// by the authenticated element alone.
fn mmr_size_of(element: &Element, path_query: &PathQuery) -> Result<u64, Error> {
    match element {
        Element::MmrTree(mmr_size, ..) => Ok(*mmr_size),
        _ => Err(invalid_proof(
            path_query,
            "MMR proof attached to non-MmrTree element".to_string(),
        )),
    }
}

/// `(total_count, height, is_bulk_append)` of a `BulkAppendTree` or
/// `CommitmentTree` element, or the error for a bulk proof attached to any
/// other element.
fn bulk_dims_of(element: &Element, path_query: &PathQuery) -> Result<(u64, u8, bool), Error> {
    match element {
        Element::BulkAppendTree(total_count, height, _) => Ok((*total_count, *height, true)),
        Element::CommitmentTree(total_count, height, _) => Ok((*total_count, *height, false)),
        _ => Err(invalid_proof(
            path_query,
            "BulkAppendTree proof attached to incompatible element".to_string(),
        )),
    }
}

/// `(count, height)` of a `DenseAppendOnlyFixedSizeTree` element, or the
/// error for a dense proof attached to any other element.
fn dense_dims_of(element: &Element, path_query: &PathQuery) -> Result<(u16, u8), Error> {
    match element {
        Element::DenseAppendOnlyFixedSizeTree(count, height, _) => Ok((*count, *height)),
        _ => Err(invalid_proof(
            path_query,
            "DenseTree proof attached to non-DenseTree element".to_string(),
        )),
    }
}

/// Where a V1 walk puts the rows it verifies.
///
/// `push_row` converts the row exactly as the matching
/// `TryFromVersioned<ProvedPathKeyOptionalValue>` conversion does, then
/// pushes it. A failed conversion pushes nothing.
pub(crate) trait RowSink {
    /// Convert `row` and append it.
    fn push_row(
        &mut self,
        row: ProvedPathKeyOptionalValue,
        grove_version: &GroveVersion,
    ) -> Result<(), Error>;
}

impl RowSink for Vec<PathKeyOptionalElementTrio> {
    /// Same as `TryFromVersioned<ProvedPathKeyOptionalValue> for
    /// PathKeyOptionalElementTrio`: deserialize the value bytes, if any.
    fn push_row(
        &mut self,
        row: ProvedPathKeyOptionalValue,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        let element = match row.value {
            Some(value) => Some(Element::deserialize(value.as_slice(), grove_version)?),
            None => None,
        };
        self.push((row.path, row.key, element));
        Ok(())
    }
}

impl RowSink for Vec<ProvedPathKeyValue> {
    /// Same as `TryFrom<ProvedPathKeyOptionalValue> for ProvedPathKeyValue`
    /// (which the versioned conversion forwards to): a row without a value
    /// is an error.
    fn push_row(
        &mut self,
        row: ProvedPathKeyOptionalValue,
        _grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        self.push(ProvedPathKeyValue::try_from(row)?);
        Ok(())
    }
}

/// What the V1 row loop does after a row: go on to the next one, or stop
/// because a limit ran out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowFlow {
    Continue,
    Break,
}

/// The read-only inputs shared by every row of one V1 layer. Shared
/// borrows only; the mutable walk state is passed separately.
///
/// `internal_query` comes before `query`: in the extracted Lean structure a
/// field named `query` would shadow the `query` namespace in the types of the
/// fields after it.
#[derive(Clone, Copy)]
struct V1LayerCtx<'a> {
    lower_layers: &'a BTreeMap<Vec<u8>, LayerProof>,
    prove_options: &'a ProveOptions,
    internal_query: &'a SinglePathSubquery<'a>,
    query: &'a PathQuery,
    current_path: &'a [&'a [u8]],
    options: &'a VerifyOptions,
    current_depth: usize,
    grove_version: &'a GroveVersion,
}

/// Whether no axis of a `ProvableCountProvableSumIndexedTree` has a
/// secondary root key, i.e. `axes.iter().all(|(_, sk)| sk.is_none())` as an
/// index loop (`true` for an empty list).
fn all_axes_unset(axes: &[(u8, Option<Vec<u8>>)]) -> bool {
    let mut i = 0;
    while i < axes.len() {
        if axes[i].1.is_some() {
            return false;
        }
        i += 1;
    }
    true
}

/// Each axis tag paired with `NULL_HASH`, in order, i.e.
/// `axes.iter().map(|(t, _)| (*t, NULL_HASH)).collect()` as an index loop.
fn null_axis_hashes(axes: &[(u8, Option<Vec<u8>>)]) -> Vec<(u8, CryptoHash)> {
    let mut hashes = Vec::with_capacity(axes.len());
    let mut i = 0;
    while i < axes.len() {
        hashes.push((axes[i].0, NULL_HASH));
        i += 1;
    }
    hashes
}

/// The first key of `lower_layers`, in ascending key order, that is not in
/// `verified_keys`; `None` when every lower layer was requested. Only the
/// error message depends on which key this is.
///
/// A plain loop rather than `keys().find(..)`, to keep closures off the
/// verifier path.
#[allow(clippy::manual_find)]
fn first_unrequested_lower_layer<'a>(
    lower_layers: &'a BTreeMap<Vec<u8>, LayerProof>,
    verified_keys: &BTreeSet<Vec<u8>>,
) -> Option<&'a Vec<u8>> {
    for key in lower_layers.keys() {
        if !verified_keys.contains(key) {
            return Some(key);
        }
    }
    None
}

impl GroveDb {
    /// Verifies a proof against a path query with the given options, returning
    /// the root hash and deserialized results.
    pub fn verify_query_with_options(
        proof: &[u8],
        path_query: &PathQuery,
        options: VerifyOptions,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Vec<PathKeyOptionalElementTrio>), Error> {
        check_grovedb_v0!(
            "verify_query_with_options",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_query_with_options
        );
        if options.absence_proofs_for_non_existing_searched_keys {
            // must have a limit
            path_query.query.limit.ok_or(Error::NotSupported(
                "limits must be set in verify_query_with_absence_proof".to_string(),
            ))?;
        }

        // Offset gate is centralized in `verify_proof_internal` — it
        // sees the envelope version and applies V0-rejects /
        // V1-relaxes uniformly across all entry points
        // (verify_query_with_options, verify_query_raw,
        // verify_query_get_parent_tree_info_with_options). Absence
        // assembly is the one consumer that can never serve an offset,
        // whatever the envelope — see
        // `PathQuery::reject_absence_proofs_with_offset`.

        // Query-shape gates run BEFORE the proof bytes are decoded: the
        // canonical decoder admits envelopes up to 256 MiB, and a query
        // this verifier cannot serve should refuse without paying for
        // (or error-shadowing) that parse. `verify_proof_internal`
        // re-checks as defense in depth.
        path_query.reject_unserved_per_instance_limits(grove_version)?;
        if options.absence_proofs_for_non_existing_searched_keys {
            path_query.reject_per_instance_limits("absence-proof verification")?;
            path_query.reject_absence_proofs_with_offset()?;
        }

        let grovedb_proof = super::decode_grovedb_proof_canonical(proof)?;

        let (root_hash, _, result) =
            Self::verify_proof_internal(&grovedb_proof, path_query, options, grove_version)?;

        Ok((root_hash, result))
    }

    /// The point of this query is to get the parent tree information which will
    /// be present because we are querying in a subtree
    pub fn verify_query_get_parent_tree_info_with_options(
        proof: &[u8],
        path_query: &PathQuery,
        options: VerifyOptions,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, TreeFeatureType, Vec<PathKeyOptionalElementTrio>), Error> {
        check_grovedb_v0!(
            "verify_query_get_parent_tree_info_with_options",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_query_get_parent_tree_info_with_options
        );

        if path_query.query.query.has_subquery() {
            return Err(Error::NotSupported(
                "getting the parent tree info is not available when using subqueries".to_string(),
            ));
        }
        if options.absence_proofs_for_non_existing_searched_keys {
            // must have a limit
            path_query.query.limit.ok_or(Error::NotSupported(
                "limits must be set in verify_query_with_absence_proof".to_string(),
            ))?;
        }

        // No offset gate here: like `verify_query_with_options`, this
        // entry defers to the envelope-aware gate in
        // `verify_proof_internal` (V0 rejects, V1 serves the
        // count-offset shape), so an offset-paginated V1 proof over a
        // provable count tree verifies here too (issue #707).

        // Pre-decode query-shape gate — see `verify_query_with_options`.
        path_query.reject_unserved_per_instance_limits(grove_version)?;
        if options.absence_proofs_for_non_existing_searched_keys {
            path_query.reject_per_instance_limits("absence-proof verification")?;
            path_query.reject_absence_proofs_with_offset()?;
        }

        let grovedb_proof = super::decode_grovedb_proof_canonical(proof)?;

        let (root_hash, tree_feature_type, result) =
            Self::verify_proof_internal(&grovedb_proof, path_query, options, grove_version)?;

        let tree_feature_type = tree_feature_type.ok_or(Error::InvalidProof(
            path_query.clone(),
            "query had no parent tree info, maybe it was for for root tree".to_string(),
        ))?;

        Ok((root_hash, tree_feature_type, result))
    }

    /// Verifies a proof against a path query and returns the root hash
    /// with raw (not deserialized) proved key-value results.
    pub fn verify_query_raw(
        proof: &[u8],
        path_query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, ProvedPathKeyValues), Error> {
        check_grovedb_v0!(
            "verify_query_raw",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_query_raw
        );
        // Pre-decode query-shape gate — see `verify_query_with_options`.
        path_query.reject_unserved_per_instance_limits(grove_version)?;

        let grovedb_proof = super::decode_grovedb_proof_canonical(proof)?;

        let (root_hash, _, result) = Self::verify_proof_raw_internal(
            &grovedb_proof,
            path_query,
            VerifyOptions {
                absence_proofs_for_non_existing_searched_keys: false,
                verify_proof_succinctness: false,
                include_empty_trees_in_result: true,
            },
            grove_version,
        )?;

        Ok((root_hash, result))
    }

    pub(crate) fn verify_proof_internal(
        proof: &GroveDBProof,
        path_query: &PathQuery,
        options: VerifyOptions,
        grove_version: &GroveVersion,
    ) -> Result<
        (
            CryptoHash,
            Option<TreeFeatureType>,
            Vec<PathKeyOptionalElementTrio>,
        ),
        Error,
    > {
        // Read-mode gate: axis / sum-budget reads are not verified by
        // this verifier. Fail closed rather than misreading such a
        // query as key selection and accepting a proof about the wrong
        // thing.
        path_query.reject_unserved_read_mode()?;

        // Per-instance limit gate: served by the V1 verifier from
        // GROVE_V4; older grove versions fail closed, and zero caps are
        // rejected as malformed. Absence-proof assembly stays rejected
        // regardless: its terminal-keys projection reports every
        // expected key the proof did not carry as absent, and which
        // keys an instance-capped walk carries is data-dependent — keys
        // beyond a cap would masquerade as proven-absent.
        path_query.reject_unserved_per_instance_limits(grove_version)?;
        if options.absence_proofs_for_non_existing_searched_keys {
            path_query.reject_per_instance_limits("absence-proof verification")?;
            // Same fail-closed reasoning for pagination: a count-offset
            // proof does not reveal which rows the offset skipped, so
            // the terminal-keys projection would report them as
            // absent. Envelope-independent, so it runs before the
            // V0/V1 offset gate below.
            path_query.reject_absence_proofs_with_offset()?;
        }

        // Offset gate centralized in `apply_count_offset_envelope_gate`:
        // V0 envelopes reject any non-zero offset (V0 is a shipped
        // wire format that never supported `SizedQuery::offset`);
        // V1 envelopes honor a non-zero offset if and only if the query
        // validates as offset-paginated. The tree-type check
        // (ProvableCountTree / ProvableCountSumTree) happens at
        // leaf-dispatch time inside `run_count_offset_layer_dispatch`.
        Self::apply_count_offset_envelope_gate(proof, path_query)?;

        match proof {
            GroveDBProof::V0(proof_v0) => {
                // The V0 verifier is a frozen wire format whose limit
                // accounting predates per-instance caps.
                path_query.reject_per_instance_limits("the V0 verifier")?;
                Self::verify_proof_v0_internal(proof_v0, path_query, options, grove_version)
            }
            GroveDBProof::V1(proof_v1) => {
                Self::verify_proof_v1_internal(proof_v1, path_query, options, grove_version)
            }
        }
    }

    /// Shared offset-envelope gate used by both `verify_proof_internal`
    /// and `verify_proof_raw_internal`. Returns `Ok(())` when the query
    /// has no non-zero offset (regular flow) or when the envelope is
    /// V1 and the query validates as offset-paginated. Returns
    /// `Error::NotSupported` when an offset is paired with a V0
    /// envelope (V0 never supported offsets and widening it would be a
    /// consensus-breaking change for shipped grove v1/v2), or whatever
    /// `validate_count_offset_paginated` returns for malformed V1
    /// offset queries. Factoring this out keeps the V0-rejects /
    /// V1-relaxes contract identical across every public entry point.
    fn apply_count_offset_envelope_gate(
        proof: &GroveDBProof,
        path_query: &PathQuery,
    ) -> Result<(), Error> {
        if !path_query.has_non_zero_offset() {
            return Ok(());
        }
        match proof {
            GroveDBProof::V0(_) => Err(Error::NotSupported(
                "offsets in path queries are not supported for proofs".to_string(),
            )),
            GroveDBProof::V1(_) => {
                path_query.validate_count_offset_paginated()?;
                Ok(())
            }
        }
    }

    fn verify_proof_v0_internal(
        proof: &GroveDBProofV0,
        query: &PathQuery,
        options: VerifyOptions,
        grove_version: &GroveVersion,
    ) -> Result<
        (
            CryptoHash,
            Option<TreeFeatureType>,
            Vec<PathKeyOptionalElementTrio>,
        ),
        Error,
    > {
        let mut result = Vec::new();
        let mut limit = query.query.limit;
        let mut last_tree_feature_type = None;
        let root_hash = Self::verify_layer_proof(
            &proof.root_layer,
            &proof.prove_options,
            query,
            &mut limit,
            &[],
            &mut result,
            &mut last_tree_feature_type,
            &options,
            0,
            grove_version,
        )?;

        if options.absence_proofs_for_non_existing_searched_keys {
            // must have a limit
            let max_results = query.query.limit.ok_or(Error::NotSupported(
                "limits must be set in verify_query_with_absence_proof".to_string(),
            ))? as usize;

            let terminal_keys = query.terminal_keys(max_results, grove_version)?;

            // convert the result set to a btree map
            let mut result_set_as_map: BTreeMap<PathKey, Option<Element>> = result
                .into_iter()
                .map(|(path, key, element)| ((path, key), element))
                .collect();
            #[cfg(feature = "proof_debug")]
            {
                println!(
                    "terminal keys are [{}] \n result set is [{}]",
                    terminal_keys
                        .iter()
                        .map(|(path, key)| format!(
                            "path: {} key: {}",
                            path_hex_to_ascii(path),
                            hex_to_ascii(key)
                        ))
                        .collect::<Vec<_>>()
                        .join(", "),
                    result_set_as_map
                        .iter()
                        .map(|((path, key), e)| {
                            let element_string = if let Some(e) = e {
                                e.to_string()
                            } else {
                                "None".to_string()
                            };
                            format!(
                                "path: {} key: {} element: {}",
                                path_hex_to_ascii(path),
                                hex_to_ascii(key),
                                element_string,
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            result = terminal_keys
                .into_iter()
                .map(|terminal_key| {
                    let element = result_set_as_map.remove(&terminal_key).flatten();
                    (terminal_key.0, terminal_key.1, element)
                })
                .collect();
        }

        Ok((root_hash, last_tree_feature_type, result))
    }

    pub(crate) fn verify_proof_raw_internal(
        proof: &GroveDBProof,
        path_query: &PathQuery,
        options: VerifyOptions,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Option<TreeFeatureType>, ProvedPathKeyValues), Error> {
        // Same fail-closed read-mode and per-instance-limit gates as
        // `verify_proof_internal`.
        path_query.reject_unserved_read_mode()?;
        path_query.reject_unserved_per_instance_limits(grove_version)?;
        if options.absence_proofs_for_non_existing_searched_keys {
            path_query.reject_per_instance_limits("absence-proof verification")?;
            path_query.reject_absence_proofs_with_offset()?;
        }

        // Same V0-rejects / V1-relaxes envelope gate as
        // `verify_proof_internal` — see `apply_count_offset_envelope_gate`.
        Self::apply_count_offset_envelope_gate(proof, path_query)?;

        match proof {
            GroveDBProof::V0(proof_v0) => {
                // See `verify_proof_internal`: V0 predates per-instance
                // caps.
                path_query.reject_per_instance_limits("the V0 verifier")?;
                Self::verify_proof_raw_internal_v0(proof_v0, path_query, options, grove_version)
            }
            GroveDBProof::V1(proof_v1) => {
                Self::verify_proof_v1_raw_internal(proof_v1, path_query, options, grove_version)
            }
        }
    }

    fn verify_proof_raw_internal_v0(
        proof: &GroveDBProofV0,
        query: &PathQuery,
        options: VerifyOptions,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Option<TreeFeatureType>, ProvedPathKeyValues), Error> {
        let mut result = Vec::new();
        let mut limit = query.query.limit;
        let mut last_tree_feature_type = None;
        let root_hash = Self::verify_layer_proof(
            &proof.root_layer,
            &proof.prove_options,
            query,
            &mut limit,
            &[],
            &mut result,
            &mut last_tree_feature_type,
            &options,
            0,
            grove_version,
        )?;
        Ok((root_hash, last_tree_feature_type, result))
    }

    fn verify_proof_v1_internal(
        proof: &GroveDBProofV1,
        query: &PathQuery,
        options: VerifyOptions,
        grove_version: &GroveVersion,
    ) -> Result<
        (
            CryptoHash,
            Option<TreeFeatureType>,
            Vec<PathKeyOptionalElementTrio>,
        ),
        Error,
    > {
        // V1 proofs do not embed ProveOptions — the verifier uses the default,
        // which closes the result-truncation attack where a malicious prover
        // could set decrease_limit_on_empty_sub_query_result to manipulate
        // how many results the verifier returns.
        let prove_options = ProveOptions::default();

        let mut result = Vec::new();
        let mut limit_state = super::V1LimitState::new(query.query.limit);
        let mut last_tree_feature_type = None;
        // This entry point serves key-selection queries only (read-mode
        // queries are gated out before reaching it), so axis outcomes
        // cannot occur; the collector exists to satisfy the walk.
        let mut axis_outcomes = Vec::new();
        let root_hash = Self::verify_layer_proof_v1(
            Self::merk_bytes_of_layer(&proof.root_layer, query)?,
            &proof.root_layer.lower_layers,
            &prove_options,
            query,
            &mut limit_state,
            None,
            &[],
            &mut result,
            &mut axis_outcomes,
            &mut last_tree_feature_type,
            &options,
            0,
            grove_version,
        )?;

        if options.absence_proofs_for_non_existing_searched_keys {
            result = Self::project_onto_terminal_keys(result, query, grove_version)?;
        }

        Ok((root_hash, last_tree_feature_type, result))
    }

    /// The absence-proof view of a V1 result set: one row per terminal key
    /// of `query` (bounded by its limit, which must be set), in
    /// `terminal_keys` order, carrying the element the walk verified for
    /// that path and key, or `None` when it verified none.
    fn project_onto_terminal_keys(
        result: Vec<PathKeyOptionalElementTrio>,
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<Vec<PathKeyOptionalElementTrio>, Error> {
        let max_results = query.query.limit.ok_or(Error::NotSupported(
            "limits must be set in verify_query_with_absence_proof".to_string(),
        ))? as usize;

        let terminal_keys = query.terminal_keys(max_results, grove_version)?;

        let mut result_set_as_map: BTreeMap<PathKey, Option<Element>> = result
            .into_iter()
            .map(|(path, key, element)| ((path, key), element))
            .collect();

        Ok(terminal_keys
            .into_iter()
            .map(|terminal_key| {
                let element = result_set_as_map.remove(&terminal_key).flatten();
                (terminal_key.0, terminal_key.1, element)
            })
            .collect())
    }

    /// Run the V1 walk for a read-mode (axis-shaped) query, returning
    /// the reconstructed root hash, the trio result set (which for axis
    /// shapes carries only absence records and any incidental
    /// key-selection rows), and the collected axis outcomes.
    ///
    /// This is the one V1 entry that does NOT gate read-mode queries
    /// out — `verify_path_query` is its only caller and applies the
    /// shape grammar (`classify`) plus the envelope gate first.
    pub(crate) fn verify_proof_v1_with_axis_outcomes(
        proof: &GroveDBProofV1,
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<
        (
            CryptoHash,
            Vec<PathKeyOptionalElementTrio>,
            Vec<AxisWalkOutcome>,
        ),
        Error,
    > {
        let prove_options = ProveOptions::default();
        let options = VerifyOptions {
            absence_proofs_for_non_existing_searched_keys: false,
            verify_proof_succinctness: true,
            include_empty_trees_in_result: false,
        };
        let mut result = Vec::new();
        let mut limit_state = super::V1LimitState::new(query.query.limit);
        let mut last_tree_feature_type = None;
        let mut axis_outcomes = Vec::new();
        let root_hash = Self::verify_layer_proof_v1(
            Self::merk_bytes_of_layer(&proof.root_layer, query)?,
            &proof.root_layer.lower_layers,
            &prove_options,
            query,
            &mut limit_state,
            None,
            &[],
            &mut result,
            &mut axis_outcomes,
            &mut last_tree_feature_type,
            &options,
            0,
            grove_version,
        )?;
        Ok((root_hash, result, axis_outcomes))
    }

    fn verify_proof_v1_raw_internal(
        proof: &GroveDBProofV1,
        query: &PathQuery,
        options: VerifyOptions,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Option<TreeFeatureType>, ProvedPathKeyValues), Error> {
        let prove_options = ProveOptions::default();

        let mut result = Vec::new();
        let mut limit_state = super::V1LimitState::new(query.query.limit);
        let mut last_tree_feature_type = None;
        // Key-selection-only entry point; see verify_proof_v1_internal.
        let mut axis_outcomes = Vec::new();
        let root_hash = Self::verify_layer_proof_v1(
            Self::merk_bytes_of_layer(&proof.root_layer, query)?,
            &proof.root_layer.lower_layers,
            &prove_options,
            query,
            &mut limit_state,
            None,
            &[],
            &mut result,
            &mut axis_outcomes,
            &mut last_tree_feature_type,
            &options,
            0,
            grove_version,
        )?;
        Ok((root_hash, last_tree_feature_type, result))
    }

    /// Count-offset leaf dispatch for `verify_layer_proof_v1`, its only
    /// caller. V0 envelopes never reach it: `apply_count_offset_envelope_gate`
    /// rejects any non-zero offset on a V0 envelope. The caller passes the
    /// unwrapped `merk_proof_bytes` and the `lower_layers_empty` flag; this
    /// helper runs `validate_count_offset_paginated`, the
    /// `verify_count_offset_on_range_proof` call, the item translation and
    /// the V1 strict-mode rejections of shapes an honest proof never
    /// surfaces.
    fn run_count_offset_layer_dispatch<S: RowSink>(
        query: &PathQuery,
        merk_proof_bytes: &[u8],
        lower_layers_empty: bool,
        current_path: &[&[u8]],
        limit_left: &mut Option<u16>,
        result: &mut S,
        grove_version: &GroveVersion,
    ) -> Result<CryptoHash, Error> {
        let inner_range = query.validate_count_offset_paginated()?.clone();
        let offset = query.query.offset.map(|o| o as u64).unwrap_or(0);
        let limit_u64 = query.query.limit.map(|l| l as u64);
        let internal_query_for_dir = match query.query_items_at_path(current_path, grove_version)? {
            Some(internal_query) => internal_query,
            None => {
                return Err(corrupted_layer_path(
                    "count-offset verify",
                    current_path,
                    query,
                ))
            }
        };

        // The validator rejects subqueries, so an honest count-offset
        // leaf proof always has empty `lower_layers`. A non-empty
        // map here means the prover attached arbitrary child layers
        // that we would otherwise silently ignore (and which the V1
        // succinctness post-pass would not catch because we
        // short-circuit before it runs).
        if !lower_layers_empty {
            return Err(invalid_proof(
                query,
                "count-offset leaf proof has unexpected lower_layers — \
                 validate_count_offset_paginated disallows subqueries, so \
                 no child layers should be present"
                    .to_string(),
            ));
        }

        let count_offset_result = grovedb_merk::proofs::query::verify_count_offset_on_range_proof(
            merk_proof_bytes,
            &inner_range,
            offset,
            limit_u64,
            internal_query_for_dir.left_to_right,
        )
        .unwrap()
        .map_err(|e| {
            invalid_proof(
                query,
                format!("count-offset merk proof failed to verify: {}", e),
            )
        })?;

        // Translate each returned item into a `ProvedPathKeyOptionalValue`.
        // Use the merk-surfaced `value_hash` and `child_hash_verified`
        // verbatim rather than recomputing `value_hash(value)` — the
        // latter is wrong for tree-flavored entries (whose committed
        // value-hash is `combine_hash(H(value), child_root)`).
        //
        // Defense-in-depth: reject any returned value whose deserialized
        // element type is one of the four shapes an honest count-offset
        // proof never surfaces. The prover-side checks in
        // `emit_count_offset_proof` already block these, so an honest
        // proof will never reach this loop with them — but a forged
        // proof might, and we don't want to silently pass tampered
        // values through. The four rejected shapes are:
        //
        //   • **NonCounted-wrapped** entries — silently dropped in
        //     normal traversal (own_count = 0) and not surfaced via
        //     the merk's `returned_items`. If one appears here, the
        //     proof was forged.
        //   • **Raw Reference / ReferenceWithSumItem** — the prover now
        //     runs a reference post-pass on this short-circuit, so an
        //     honest proof surfaces the dereferenced TARGET here, never
        //     the reference itself. An unresolved one is malformed.
        //   • **Non-empty tree** — V1 strict-mode would require a
        //     `KVValueHashFeatureTypeWithChildHash` proof node here;
        //     accepting one without that would silently bypass the
        //     child-hash invariant the regular flow enforces.
        //   • **Any other unresolved row that is not a simple item or
        //     tree** — today the backward-references items. See the
        //     allowlist in the loop below.
        for item in count_offset_result.returned_items.iter() {
            // Reject non-Element bytes — a forged proof might surface raw
            // bytes that don't deserialize as any known Element type. The
            // count-offset prover only emits real Element bytes, so a
            // deserialization failure here means a malformed or forged
            // payload. Previously this branch silently fell through to the
            // `result.push` below, surfacing the unparsed bytes to the
            // caller.
            let elem = Element::deserialize(item.value.as_slice(), grove_version).map_err(|e| {
                invalid_proof(
                    query,
                    format!(
                        "count-offset paginated proof returned non-Element bytes at \
                         key {}: {e}",
                        hex::encode(&item.key)
                    ),
                )
            })?;
            // NonCounted-wrapped values are checked **before**
            // unwrapping via `into_underlying`, since the wrapper
            // itself is the rejected shape. The merk-level prover
            // already refuses to emit NonCounted entries of the proved
            // tree as value-bearing nodes, so an honest proof can never
            // surface one of the tree's OWN entries wrapped. Reject as
            // `InvalidProof` (forgery) rather than `NotSupported` to
            // make the distinction visible.
            //
            // A row resolved through a reference is the exception: a
            // reference commits to its terminal's stored bytes verbatim
            // (wrapper included — see `follow_reference_as_stored`), so
            // a target that lives wrapped in a count-bearing tree
            // legitimately surfaces here wrapped. The merk verifier
            // flags those rows, and the wrapper is looked through below
            // like everywhere else.
            if elem.is_non_counted() && !item.resolved_from_reference {
                return Err(invalid_proof(
                    query,
                    format!(
                        "count-offset paginated proofs do not surface \
                         NonCounted-wrapped entries in returned items — proof at \
                         key {} appears forged",
                        hex::encode(&item.key)
                    ),
                ));
            }
            let inner = elem.into_underlying();
            if inner.is_non_empty_tree() {
                return Err(Error::NotSupported(format!(
                    "count-offset paginated proofs do not yet support \
                     non-empty tree return values (key {})",
                    hex::encode(&item.key)
                )));
            }
            // A RAW reference must not reach the caller: the prover's
            // post-pass rewrites reference rows into resolved-value nodes,
            // so by this point `item.value` is the dereferenced TARGET and
            // `item.reference_element_hash` records that a reference was
            // resolved. Seeing an unresolved reference here means the proof
            // skipped the post-pass — reject rather than surface it.
            if inner.is_reference() {
                return Err(invalid_proof(
                    query,
                    format!(
                        "count-offset paginated proof surfaced a raw \
                         Reference / ReferenceWithSumItem at key {} — the prover's \
                         reference post-pass rewrites these into resolved-value \
                         nodes, so an unresolved reference here is malformed",
                        hex::encode(&item.key)
                    ),
                ));
            }
            // Allowlist the element types an unresolved row may hold
            // (defense in depth behind the merk-level allowlist in
            // `classify_self`). A row not resolved through a reference
            // is one of the count tree's own entries. An honest proof
            // surfaces only two kinds of entry here: simple items, on
            // `KVCount` / `KVCountSum`, which recompute `H(value)`, and
            // empty subtrees, checked against their committed hash
            // below. Any other type is a forgery. On a
            // `KVValueHashFeatureType` node the hash does not bind the
            // value bytes; on a `KVCount` / `KVCountSum` node the bytes
            // cannot match the committed root, because the tree never
            // stores that type. Today that means the backward-references
            // items, which Provable* count trees refuse at insert. Rows
            // resolved through a reference are exempt: their value is
            // the target, which the `KVRefValueHash*` combine binds.
            if !item.resolved_from_reference {
                let element_type = inner.element_type();
                if !(element_type.has_simple_value_hash() || element_type.is_tree()) {
                    return Err(invalid_proof(
                        query,
                        format!(
                            "count-offset paginated proof surfaced an element of type '{}' at key {}; \
                             a count tree's own rows are simple items or empty trees, and \
                             any other type rides on a node whose hash does not bind its \
                             value bytes, so the row is forged",
                            element_type.as_str(),
                            hex::encode(&item.key)
                        ),
                    ));
                }
            }
            // Empty-tree value-hash equality check (defense-in-depth on
            // top of the merk-level KV→KVValueHash forgery guard).
            //
            // For an empty Merk subtree (`Tree(None, _)`, `SumTree(None,
            // ..)`, etc.), the committed value-hash is
            // `combine_hash(value_hash(value), NULL_HASH)`. The
            // proof-carried `item.value_hash` is structurally trusted by
            // the merk-level chain check (it's what the reconstruction
            // hashes); if it doesn't actually equal the expected
            // composition, the bytes were swapped without the proof
            // catching it. Recompute and compare.
            //
            // Skipped for non-tree elements (Items / SumItems) — their
            // value-hash IS `value_hash(value)` and is already enforced
            // by the merk-level KVCount path, which recomputes it via
            // `compute_value_hash` rather than trusting the proof.
            //
            // Skipped for non-empty trees and references — those are
            // already rejected above as NotSupported.
            if inner.is_any_tree() && !inner.is_non_empty_tree() {
                use grovedb_merk::tree::{
                    axes_digest, combine_hash, combine_hash_three, value_hash, NULL_HASH,
                };
                let value_h = value_hash(item.value.as_slice()).unwrap();
                // Indexed trees (PCIT / PSIT / PCPSIT) commit a three-input
                // combine_hash_three(H(value), NULL_HASH, second) even when
                // empty — `second` is NULL_HASH for PCIT/PSIT and
                // axes_digest(zero_axes) for PCPSIT. The two-input
                // combine_hash form applies to every other empty tree. Mirror
                // the terminal-path fix so honest empty-indexed returns pass.
                let expected = match &inner {
                    Element::ProvableCountIndexedTree(None, None, 0, _)
                    | Element::ProvableSumIndexedTree(None, None, 0, _) => {
                        combine_hash_three(&value_h, &NULL_HASH, &NULL_HASH).unwrap()
                    }
                    Element::ProvableCountProvableSumIndexedTree(None, 0, 0, axes, _)
                        if all_axes_unset(axes) =>
                    {
                        let zero_axes: Vec<(u8, grovedb_merk::CryptoHash)> = null_axis_hashes(axes);
                        let digest = axes_digest(&zero_axes).unwrap();
                        combine_hash_three(&value_h, &NULL_HASH, &digest).unwrap()
                    }
                    _ => combine_hash(&value_h, &NULL_HASH).unwrap(),
                };
                if expected != item.value_hash {
                    return Err(invalid_proof(
                        query,
                        format!(
                            "count-offset paginated proof: empty-tree return at key {} \
                             has value_hash {} but expected {} — possible KV→KVValueHash \
                             forgery",
                            hex::encode(&item.key),
                            hex::encode(item.value_hash),
                            hex::encode(expected),
                        ),
                    ));
                }
            }
            let proved_key_optional_value = grovedb_merk::proofs::query::ProvedKeyOptionalValue {
                key: item.key.clone(),
                value: Some(item.value.clone()),
                proof: item.value_hash,
                child_hash_verified: item.child_hash_verified,
            };
            let path_key_optional_value = ProvedPathKeyOptionalValue::from_proved_key_value(
                current_path.iter().map(|p| p.to_vec()).collect(),
                proved_key_optional_value,
            );
            result.push_row(path_key_optional_value, grove_version)?;
            limit_left
                .iter_mut()
                .for_each(|limit| *limit = limit.saturating_sub(1));
        }
        Ok(count_offset_result.root_hash)
    }

    /// Destructure a layer's proof body, rejecting non-Merk shapes. Callers
    /// pass the resulting slice to `verify_layer_proof_v1`.
    fn merk_bytes_of_layer<'a>(
        layer: &'a LayerProof,
        query: &PathQuery,
    ) -> Result<&'a [u8], Error> {
        match &layer.merk_proof {
            ProofBytes::Merk(bytes) => Ok(bytes.as_slice()),
            ProofBytes::MMR(_)
            | ProofBytes::BulkAppendTree(_)
            | ProofBytes::DenseTree(_)
            | ProofBytes::CommitmentTree(_)
            | ProofBytes::CountIndexedTree(_)
            | ProofBytes::IndexedTreeTerminal(_)
            | ProofBytes::IndexedTreeAxisDescent(_)
            | ProofBytes::SumBudgetWindow(_) => Err(invalid_proof(
                query,
                "Expected Merk proof at this layer, got non-Merk proof type".to_string(),
            )),
        }
    }

    /// Verify one axis-ordered descent layer: the embedded
    /// (V1-envelope) counterpart of the standalone indexed-axis
    /// envelopes.
    ///
    /// Trust chain, strictly downward:
    /// 1. The secondary proof is verified for the query's traversal,
    ///    yielding the entries/aggregate AND the secondary's
    ///    reconstructed root hash — the attestation is **recomputed**,
    ///    never accepted as raw bytes.
    /// 2. The proved element's family is checked against the queried
    ///    axis, and the third `combine_hash_three` input is derived
    ///    (recomputed secondary root for PCIT/PSIT; `axes_digest` over
    ///    the payload's other-axes list plus the recomputed root for
    ///    PCPSIT). Forged `primary_root_hash` / `other_axes` fail the
    ///    binding below because they are part of its preimage.
    /// 3. `combine_hash_three(H(value), primary_root, digest)` must
    ///    equal the parent-committed `value_hash` the surrounding walk
    ///    already verified.
    ///
    /// On success the traversal's verified answer is pushed into
    /// `axis_outcomes` keyed by the indexed tree's full path.
    #[allow(clippy::too_many_arguments)]
    fn verify_axis_descent_layer(
        lower_layer: &LayerProof,
        axis_query: &grovedb_merk::proofs::query::AxisQuery,
        element_value_bytes: &[u8],
        parent_committed_hash: &CryptoHash,
        path: &[&[u8]],
        axis_outcomes: &mut Vec<AxisWalkOutcome>,
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        use grovedb_merk::proofs::query::{
            verify_aggregate_count_on_range_proof, verify_aggregate_sum_on_range_proof,
            verify_count_offset_on_range_proof, AxisTraversal, QueryItem as MerkQueryItemForRange,
        };

        use crate::operations::proof::indexed_axis::verify::{
            count_aggregate_inner_range, decode_axis_entries_from_count_offset_items,
            decode_axis_entries_from_result_set, recompute_axis_binding_digest,
            sum_aggregate_inner_range,
        };

        // Envelope gate: the axis descent is a V4 acceptance rule.
        if grove_version
            .grovedb_versions
            .operations
            .proof
            .axis_descent_in_v1_envelope
            != 1
        {
            return Err(Error::NotSupported(
                "axis-ordered descents in the V1 proof envelope are not accepted at this \
                 grove version"
                    .to_string(),
            ));
        }

        let ProofBytes::IndexedTreeAxisDescent(payload_bytes) = &lower_layer.merk_proof else {
            return Err(invalid_proof(
                query,
                "the query node at this indexed tree is an axis read; its lower layer must \
                 carry ProofBytes::IndexedTreeAxisDescent"
                    .to_string(),
            ));
        };
        if !lower_layer.lower_layers.is_empty() {
            return Err(invalid_proof(
                query,
                "an axis descent is terminal and must not carry further lower layers".to_string(),
            ));
        }
        let payload = AxisDescentProof::decode_canonical(payload_bytes)?;
        let axis = axis_query.axis;
        if payload.axis_tag != axis.tag() {
            return Err(invalid_proof(
                query,
                format!(
                    "axis descent payload claims axis tag {} but the query reads axis {:?}",
                    payload.axis_tag, axis
                ),
            ));
        }
        if payload.rank.is_some()
            && !matches!(axis_query.traversal, AxisTraversal::RankOfKey { .. })
        {
            return Err(invalid_proof(
                query,
                "axis descent payload carries a rank but the traversal is not RankOfKey"
                    .to_string(),
            ));
        }

        // 1. Verify the secondary proof for the query's traversal,
        //    recomputing the secondary root hash.
        let (secondary_root_hash, walk_result) = match &axis_query.traversal {
            AxisTraversal::RankedPage { k, offset } => {
                let inner_range = MerkQueryItemForRange::RangeFull(std::ops::RangeFull);
                let res = verify_count_offset_on_range_proof(
                    &payload.secondary_proof,
                    &inner_range,
                    *offset,
                    Some(*k as u64),
                    !axis_query.descending,
                )
                .unwrap()
                .map_err(|e| {
                    invalid_proof(
                        query,
                        format!("axis descent: secondary count-offset proof failed: {e}"),
                    )
                })?;
                let entries = decode_axis_entries_from_count_offset_items(
                    axis,
                    &res.returned_items,
                    &payload.target_chains,
                    grove_version,
                )?;
                (
                    res.root_hash,
                    AxisWalkResult::Entries {
                        entries,
                        skipped: Some(res.skipped),
                    },
                )
            }
            AxisTraversal::RankOfKey { key } => {
                let Some(rank) = payload.rank else {
                    return Err(invalid_proof(
                        query,
                        "axis descent for a rank-of-key traversal must carry the rank".to_string(),
                    ));
                };
                let inner_range = MerkQueryItemForRange::RangeFull(std::ops::RangeFull);
                let res = verify_count_offset_on_range_proof(
                    &payload.secondary_proof,
                    &inner_range,
                    rank,
                    Some(1),
                    !axis_query.descending,
                )
                .unwrap()
                .map_err(|e| {
                    invalid_proof(
                        query,
                        format!("axis descent: rank count-offset proof failed: {e}"),
                    )
                })?;
                // The count commitments attest the skip; the claimed
                // rank must be fully realized (not merely "walk ran
                // out"), and the single yielded entry must be the
                // queried key.
                if res.skipped != rank {
                    return Err(invalid_proof(
                        query,
                        format!(
                            "axis descent: claimed rank {rank} but the count commitments \
                             attest {} skipped entries",
                            res.skipped
                        ),
                    ));
                }
                let entries = decode_axis_entries_from_count_offset_items(
                    axis,
                    &res.returned_items,
                    &payload.target_chains,
                    grove_version,
                )?;
                let yielded_key = entries.first_original_key().map(|k| k.to_vec());
                if yielded_key.as_deref() != Some(key.as_slice()) {
                    return Err(invalid_proof(
                        query,
                        format!(
                            "axis descent: the entry at rank {rank} is {:?}, not the queried \
                             key {}",
                            yielded_key.map(hex::encode),
                            hex::encode(key),
                        ),
                    ));
                }
                (res.root_hash, AxisWalkResult::Rank { rank })
            }
            AxisTraversal::Bounded { limit, .. } => {
                if payload.secondary_proof.is_empty() {
                    // Empty-secondary convention (mirrors the
                    // count-offset and aggregate-on-range verifiers):
                    // empty proof bytes resolve to a NULL_HASH secondary
                    // root and no entries. Sound because the binding
                    // below only passes when the element genuinely
                    // commits an empty secondary — claiming emptiness
                    // for a populated one needs a Blake3 second
                    // preimage.
                    (
                        NULL_HASH,
                        AxisWalkResult::Entries {
                            entries: AxisEntries::empty_for_axis(axis),
                            skipped: None,
                        },
                    )
                } else {
                    let secondary_query =
                        crate::query::axis_lowering::axis_bounded_merk_query(axis_query)?;
                    let left_to_right = secondary_query.left_to_right;
                    // Strict mode (#863): the secondary stream must be
                    // encoded in the family of the direction it is
                    // walked in, or an upright stream handed to a
                    // descending axis read would fill the page from the
                    // wrong end of the range. The strict value checks
                    // that come with it are moot here — the axis
                    // decoders consume only `proved.key`, bound into the
                    // recomputed secondary root — but harmless.
                    let (root, res) = secondary_query
                        .execute_proof(
                            &payload.secondary_proof,
                            Some(*limit),
                            left_to_right,
                            PROOF_VERSION_LATEST,
                        )
                        .unwrap()
                        .map_err(|e| {
                            invalid_proof(
                                query,
                                format!("axis descent: secondary range proof failed: {e}"),
                            )
                        })?;
                    let entries = decode_axis_entries_from_result_set(
                        axis,
                        &res.result_set,
                        &payload.target_chains,
                        grove_version,
                    )?;
                    (
                        root,
                        AxisWalkResult::Entries {
                            entries,
                            skipped: None,
                        },
                    )
                }
            }
            AxisTraversal::AggregateOverValueRange { lo, hi, fold } => {
                // The byte range follows the AXIS; the walker follows
                // the FOLD — the same split the prover makes, through
                // the same range reconstructors, so the two sides
                // cannot drift on clamping, degenerate, or
                // out-of-domain shapes.
                let inner_range = match axis {
                    grovedb_merk::proofs::query::IndexAxis::Count => {
                        count_aggregate_inner_range(*lo, *hi)
                    }
                    grovedb_merk::proofs::query::IndexAxis::Sum => {
                        sum_aggregate_inner_range(*lo, *hi)
                    }
                    grovedb_merk::proofs::query::IndexAxis::Avg => {
                        return Err(invalid_proof(
                            query,
                            "axis descent: value-range aggregates are not defined for the \
                             Avg axis"
                                .to_string(),
                        ));
                    }
                };
                match fold {
                    grovedb_merk::proofs::query::AggregateFold::Population => {
                        let (root, count) = verify_aggregate_count_on_range_proof(
                            &payload.secondary_proof,
                            &inner_range,
                        )
                        .unwrap()
                        .map_err(|e| {
                            invalid_proof(
                                query,
                                format!("axis descent: population proof failed: {e}"),
                            )
                        })?;
                        (
                            root,
                            AxisWalkResult::Aggregate {
                                value: count as i128,
                            },
                        )
                    }
                    grovedb_merk::proofs::query::AggregateFold::Total => {
                        let (root, sum) = verify_aggregate_sum_on_range_proof(
                            &payload.secondary_proof,
                            &inner_range,
                        )
                        .unwrap()
                        .map_err(|e| {
                            invalid_proof(query, format!("axis descent: total proof failed: {e}"))
                        })?;
                        (root, AxisWalkResult::Aggregate { value: sum as i128 })
                    }
                }
            }
        };

        // 2. Family check + third-input digest, with the RECOMPUTED
        //    secondary root.
        let digest = recompute_axis_binding_digest(
            element_value_bytes,
            axis,
            &secondary_root_hash,
            &payload.other_axes_root_hashes,
            payload.target_is_pcpsit,
            "axis descent",
        )?;

        // 3. Parent binding.
        let combined = combine_hash_three(
            value_hash(element_value_bytes).value(),
            &payload.primary_root_hash,
            &digest,
        )
        .value()
        .to_owned();
        if parent_committed_hash != &combined {
            return Err(invalid_proof(
                query,
                format!(
                    "axis descent binding mismatch: parent committed {}, recomputed {}",
                    hex::encode(parent_committed_hash),
                    hex::encode(combined),
                ),
            ));
        }

        axis_outcomes.push(AxisWalkOutcome {
            path: path.iter().map(|segment| segment.to_vec()).collect(),
            result: walk_result,
        });
        Ok(())
    }

    /// Verify one sum-budget window layer: execute the carried Merk
    /// proof with the query's own items and **replay the budget fold**
    /// — the exact per-element arithmetic of the trusted read engine,
    /// with the provable semantics (skip non-sum elements, skip
    /// references) — over the proved elements in walk order.
    ///
    /// The replay attests the stop condition: a window that continues
    /// past a fired stop, claims exhaustion while a stop fired, or
    /// claims a stop that never fired is rejected. Returns the window
    /// proof's reconstructed root hash, which the surrounding walk
    /// binds through the ordinary `combine_hash(H(value), child_root)`
    /// parent check.
    ///
    /// `sum_budget_window_gate` runs first and returns the budget read the
    /// query asks for at `path`; only then does `sum_budget_window_body`
    /// decode and replay the window.
    fn verify_sum_budget_window_layer(
        payload_bytes: &[u8],
        path: &[&[u8]],
        axis_outcomes: &mut Vec<AxisWalkOutcome>,
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<CryptoHash, Error> {
        let budget = Self::sum_budget_window_gate(path, query, grove_version)?;
        Self::sum_budget_window_body(
            budget,
            payload_bytes,
            path,
            axis_outcomes,
            query,
            grove_version,
        )
    }

    /// The acceptance gate of a sum-budget window layer: the grove version
    /// must accept sum-budget windows in the V1 envelope, and the query
    /// node at `path` must be a sum-budget read, whose budget is returned.
    fn sum_budget_window_gate(
        path: &[&[u8]],
        path_query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<SumBudgetRead, Error> {
        // Envelope gate: sum-budget windows are a V4 acceptance rule.
        if grove_version
            .grovedb_versions
            .operations
            .proof
            .sum_budget_in_v1_envelope
            != 1
        {
            return Err(Error::NotSupported(
                "sum-budget windows in the V1 proof envelope are not accepted at this grove \
                 version"
                    .to_string(),
            ));
        }
        let Some(budget) = path_query.sum_budget_read_at_path(path).copied() else {
            return Err(invalid_proof(
                path_query,
                "the proof carries a sum-budget window at a path whose query node is not a \
                 sum-budget read"
                    .to_string(),
            ));
        };
        Ok(budget)
    }

    /// The body of a sum-budget window layer, run after
    /// `sum_budget_window_gate` accepted it with `budget`: decode the
    /// window, execute its Merk proof, replay the budget fold, record the
    /// outcome in `axis_outcomes`, and return the window root.
    fn sum_budget_window_body(
        budget: SumBudgetRead,
        payload_bytes: &[u8],
        path: &[&[u8]],
        axis_outcomes: &mut Vec<AxisWalkOutcome>,
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<CryptoHash, Error> {
        use crate::operations::proof::SumBudgetWindowProof;

        let node = &query.query.query;

        let payload = SumBudgetWindowProof::decode_canonical(payload_bytes)?;

        // Execute the window proof with the query's own items and
        // direction. When the prover claims exhaustion the proof must
        // stand WITHOUT a limit (the range end is proven); otherwise it
        // is limited to the claimed window.
        let mut window_query = Query::new_with_direction(node.left_to_right);
        window_query.items = node.items.clone();
        let execute_limit = if payload.exhausted {
            None
        } else {
            Some(payload.window_len)
        };
        let (window_root, window_result) = window_query
            .execute_proof(
                &payload.merk_proof,
                execute_limit,
                node.left_to_right,
                PROOF_VERSION_LATEST,
            )
            .unwrap()
            .map_err(|e| {
                invalid_proof(
                    query,
                    format!("sum-budget window proof failed to execute: {e}"),
                )
            })?;

        // Present rows only, in walk order, each kept WITH its binding
        // evidence (the merk-surfaced value hash and whether a child hash
        // was verified); absent keys (value None) are covered by the
        // proof but never scanned by the engine.
        let window: Vec<(&Vec<u8>, &Vec<u8>, &CryptoHash, bool)> = window_result
            .result_set
            .iter()
            .filter_map(|row| {
                row.value
                    .as_ref()
                    .map(|value| (&row.key, value, &row.proof, row.child_hash_verified))
            })
            .collect();
        if window.len() != payload.window_len as usize {
            return Err(invalid_proof(
                query,
                format!(
                    "sum-budget window claims {} scanned elements but the proof yields {}",
                    payload.window_len,
                    window.len()
                ),
            ));
        }

        // Replay the budget fold with the engine's exact arithmetic.
        let mut remaining: i64 = i64::try_from(budget.sum_limit)
            .map_err(|_| Error::InvalidQuery("sum-budget limit must fit in i64"))?;
        let mut matches_left = budget.match_limit;
        let global_cap = grove_version
            .grovedb_versions
            .query_limits
            .max_aggregate_sum_query_elements_scanned;
        let mut scanned: u16 = 0;
        let mut matches: Vec<(Vec<u8>, i64)> = Vec::new();
        let mut hard_cap_tripped = false;

        for (key, value_bytes, row_value_hash, child_hash_verified) in &window {
            // Pre-element conditions (the engine's loop guards): a
            // window that continues past a fired stop hides where the
            // walk really ended.
            if remaining <= 0 || matches_left == Some(0) || hard_cap_tripped {
                return Err(invalid_proof(
                    query,
                    "sum-budget window continues past its stop condition".to_string(),
                ));
            }
            // Binding evidence (#870): the fold/skip decision below reads
            // the element bytes, so they must be the bytes the root
            // commits. An item row is bound when the merk verifier hashed
            // its value itself (`H(value) == value_hash`); a composite row
            // (tree, reference) is bound only when the merk verifier
            // checked `combine_hash(H(value), child_hash) == value_hash`
            // on a node carrying the child hash; a backward-references
            // item row (`KVBackwardsReferencesValueHash`) is bound the same
            // way, the merk verifier having recomputed
            // `combine_hash(H(stripped), referrer_list_hash)` into the root.
            // A bare `KVValueHash` row satisfies neither: its bytes are
            // free for a prover to rewrite under a genuine root, which is
            // exactly how a sum item could be disguised as a tree and
            // dropped.
            let simply_bound = value_hash(value_bytes).value() == *row_value_hash;
            if !simply_bound && !*child_hash_verified {
                return Err(invalid_proof(
                    query,
                    format!(
                        "sum-budget window row {} presents element bytes the proof does not \
                         bind: a composite row must carry its child hash",
                        hex::encode(key)
                    ),
                ));
            }
            scanned = scanned.saturating_add(1);
            if scanned > global_cap {
                // The engine counts the tripping element but does not
                // process it.
                hard_cap_tripped = true;
                continue;
            }
            let element = Element::deserialize(value_bytes, grove_version)?;
            // Provable fold semantics: references and non-sum elements
            // are skipped (they cannot be replayed / do not contribute).
            // Their bytes are authenticated above, so the skip is too.
            if element.is_reference() || !element.is_sum_item() {
                continue;
            }
            // A plain sum item commits the plain hash of its bytes, and
            // the merk verifier refuses item elements on child-hash nodes
            // at proof version 1, so a row that reaches the fold is either
            // simply bound or — for the backward-references sum items —
            // bound through the recomputed referrer-list combination.
            debug_assert!(
                simply_bound || *child_hash_verified,
                "a folded sum item row must be bound by its own hash or a recomputed \
                 combined hash"
            );
            let value = match element.into_underlying() {
                Element::SumItem(value, _)
                | Element::ItemWithSumItem(_, value, _)
                | Element::SumItemWithBackwardsReferences(value, _, _)
                | Element::ItemWithSumItemWithBackwardsReferences(_, value, _, _) => value,
                _ => {
                    return Err(invalid_proof(
                        query,
                        "sum-budget window element passed the sum-item check but carries no \
                         sum value"
                            .to_string(),
                    ));
                }
            };
            matches.push(((*key).clone(), value));
            if let Some(limit) = matches_left.as_mut() {
                *limit = limit.saturating_sub(1);
            }
            remaining = remaining.saturating_sub(value);
        }

        let stop_fired = remaining <= 0 || matches_left == Some(0) || hard_cap_tripped;
        if payload.exhausted && stop_fired {
            return Err(invalid_proof(
                query,
                "sum-budget window claims exhaustion but a stop condition fired within it"
                    .to_string(),
            ));
        }
        if !payload.exhausted && !stop_fired {
            return Err(invalid_proof(
                query,
                "sum-budget window claims a stop condition but none fired within it — the \
                 window is short of the real stop"
                    .to_string(),
            ));
        }

        let stop = if payload.exhausted {
            SumBudgetStop::Exhausted
        } else if remaining <= 0 {
            SumBudgetStop::BudgetReached
        } else if matches_left == Some(0) {
            SumBudgetStop::MatchLimitReached
        } else {
            SumBudgetStop::HardScanCapReached
        };
        let total = matches
            .iter()
            .map(|(_, value)| *value)
            .fold(0i64, |acc, v| acc.saturating_add(v));

        axis_outcomes.push(AxisWalkOutcome {
            path: path.iter().map(|segment| segment.to_vec()).collect(),
            result: AxisWalkResult::SumBudget {
                matches,
                total,
                stop,
            },
        });
        Ok(window_root)
    }

    /// Derive a Merk layer's root hash without reporting any of its rows.
    ///
    /// Used when a subset verification stops at a tree ELEMENT that the proof
    /// descended into: the parent still has to bind the element bytes with
    /// `combine_hash(H(value), child_root)`, but none of the child layer's rows
    /// belong in this query's result set. An empty query walks the proof for
    /// its root and matches nothing, so no row is produced and no limit is
    /// consumed.
    ///
    /// The layer's own `lower_layers` are deliberately not descended into. The
    /// root returned here is computed from this layer's nodes alone, whose
    /// value hashes already commit to everything beneath them; the deeper
    /// layers exist only to authenticate rows that are not being reported.
    fn merk_layer_root_hash(
        merk_proof_bytes: &[u8],
        query: &PathQuery,
    ) -> Result<CryptoHash, Error> {
        // The layer was emitted in the direction of the query that
        // generated the proof, which this (subset) query does not know.
        // No row is reported, so the direction carries no semantics
        // here; it only has to match the stream's own family for the
        // #863 orientation check, which still refuses a mixed stream.
        let left_to_right = grovedb_merk::proofs::query::proof_stream_direction(merk_proof_bytes)
            .map_err(|e| {
                invalid_proof(
                    query,
                    format!("Invalid V1 lower layer proof (root derivation): {}", e),
                )
            })?
            .unwrap_or(true);
        let (root_hash, _) = Query::new_with_direction(left_to_right)
            .execute_proof(merk_proof_bytes, None, left_to_right, PROOF_VERSION_LATEST)
            .unwrap()
            .map_err(|e| {
                invalid_proof(
                    query,
                    format!("Invalid V1 lower layer proof (root derivation): {}", e),
                )
            })?;
        Ok(root_hash)
    }

    /// Takes the layer's Merk proof BYTES and its lower layers separately
    /// rather than a `&LayerProof`.
    ///
    /// The indexed-tree descent needs to recurse with a different proof body
    /// (the primary proof carved out of the indexed envelope) but the SAME
    /// descendant layers. Passing a `&LayerProof` forced it to synthesize one,
    /// which meant deep-copying the whole `lower_layers` map at every nesting
    /// level — O(depth x payload) memory on a proof whose depth and payload an
    /// untrusted peer chooses. Splitting the parameters lets that descent pass
    /// a borrowed slice and a borrowed map, so nothing is copied.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_layer_proof_v1<S: RowSink>(
        merk_proof_bytes: &[u8],
        lower_layers: &BTreeMap<Vec<u8>, LayerProof>,
        prove_options: &ProveOptions,
        path_query: &PathQuery,
        limit_state: &mut super::V1LimitState,
        inherited_instance: Option<u16>,
        current_path: &[&[u8]],
        result: &mut S,
        axis_outcomes: &mut Vec<AxisWalkOutcome>,
        last_parent_tree_type: &mut Option<TreeFeatureType>,
        options: &VerifyOptions,
        current_depth: usize,
        grove_version: &GroveVersion,
    ) -> Result<CryptoHash, Error> {
        if current_depth > super::MAX_PROOF_DEPTH {
            return Err(invalid_proof(
                path_query,
                "proof verification exceeded maximum depth limit".to_string(),
            ));
        }
        // Count-offset paginated dispatch (v1 verify). Fires when:
        //   - we're at the leaf level (current_path == query.path), and
        //   - the path query has a non-zero offset, and
        //   - it validates as count-offset-paginated (syntactic gate
        //     already passed at the top entry, so this should
        //     always succeed for honest callers but we double-check
        //     to surface invariant violations cleanly).
        //
        // On match: route to the merk-level
        // `verify_count_offset_on_range_proof`, convert the returned
        // items into `ProvedPathKeyOptionalValue`s the rest of the
        // verifier pipeline expects, and return the leaf merk's root
        // hash so the parent layer's `combine_hash(H(value),
        // lower_hash)` chain check matches.
        if current_path.len() == path_query.path.len() && path_query.has_non_zero_offset() {
            // Count-offset queries reject per-instance limits, so the
            // effective budget here is just the global one; the temp
            // keeps the dispatch's own accounting and reports the
            // consumed slots back through the shared state.
            let mut count_offset_limit = limit_state.effective_layer_limit(inherited_instance);
            let count_offset_before = count_offset_limit;
            let dispatch_result = Self::run_count_offset_layer_dispatch(
                path_query,
                merk_proof_bytes,
                lower_layers.is_empty(),
                current_path,
                &mut count_offset_limit,
                result,
                grove_version,
            );
            let consumed = count_offset_before
                .unwrap_or(0)
                .saturating_sub(count_offset_limit.unwrap_or(0));
            limit_state.charge_rows(consumed);
            return dispatch_result;
        }

        let internal_query = match path_query.query_items_at_path(current_path, grove_version)? {
            Some(internal_query) => internal_query,
            None => return Err(corrupted_layer_path("verify v1", current_path, path_query)),
        };

        // This frame's per-instance budget — fresh for this subtree,
        // bounded by every enclosing cap; `None` throughout on queries
        // without per-instance limits. Mirrors `prove_subqueries_v1`.
        let mut frame_instance =
            super::V1LimitState::min_caps(inherited_instance, internal_query.instance_limit);

        // Which direction this layer's op stream is encoded in.
        //
        // For a real query node the direction is query semantics — it
        // decides which end of a range fills a limit — so it comes from
        // the query and nothing else. A *synthesized* path-component
        // level is different: `query_items_at_path` manufactures it for
        // every path component above the query's own path (and for
        // positions inside a `subquery_path`), its item list is exactly
        // one `QueryItem::Key`, and its `left_to_right` is a fixed
        // placeholder. The generating query — which is what chose the
        // op family the prover emitted at this path — is not
        // recoverable from a subset query, so no fixed value is right:
        // a merged descending query emits this layer inverted, and
        // running the ascending bound-witness machinery over an
        // inverted stream is wrong in both directions (it rejects
        // honest proofs, and worse, it can read an absence out of a
        // stream that proves presence).
        //
        // So for a one-key synthesized level, read the orientation off
        // the proof's own op family. That is not trusting an
        // attacker-chosen parameter: `execute` checks, per op, that
        // upright pushes ascend and inverted pushes descend, so a
        // stream cannot claim an orientation it does not have — and
        // with a single `Key` item the direction cannot reorder,
        // truncate or extend the answer either way. Everything that
        // binds the result stays where it was: the reconstructed root
        // hash still has to match what the parent layer committed, and
        // `QueryItem::contains` still gates every returned key.
        //
        // For every other level the direction is the query's, and
        // `execute_proof` itself (#863) refuses a stream that is not
        // homogeneous in that direction's op family — an upright
        // stream walked descending, or a mixed stream that rebuilds
        // the honest tree in a non-monotonic visit order, would
        // otherwise read an absence, or a page from the wrong end of
        // the range, out of an authentic root hash. The same check
        // runs on a synthesized level, where the direction read off
        // the stream trivially matches it (`proof_stream_direction`
        // already refuses a mixed stream).
        let single_key_synthesized_level = is_single_key_synthesized_level(&internal_query);
        let left_to_right = if single_key_synthesized_level {
            grovedb_merk::proofs::query::proof_stream_direction(merk_proof_bytes)
                .map_err(|e| {
                    invalid_proof(
                        path_query,
                        format!("Invalid V1 proof op stream at path component layer: {}", e),
                    )
                })?
                // An op-less stream carries no orientation; `execute`
                // rejects it a moment later for having no root.
                .unwrap_or(internal_query.left_to_right)
        } else {
            internal_query.left_to_right
        };

        let level_query = Query {
            items: internal_query.items.to_vec(),
            left_to_right,
            ..Default::default()
        };

        // Mirror of the prover's per-layer limit rule: a pure terminal
        // layer (no subquery branches) runs under the tighter of the
        // global and instance budgets — its merk rows ARE result rows.
        // A layer with subquery branches runs under the global budget
        // only: the instance chain budgets descendant ROWS, not
        // children, and must not truncate which children appear (an
        // empty child consumes no instance budget, so a later populated
        // child may still owe rows). Prover and verifier must derive
        // the identical value here or honest proofs fail the
        // more-data-than-limit check.
        let layer_limit = v1_layer_limit(&internal_query.has_subquery, limit_state, frame_instance);

        let (root_hash, merk_result) = level_query
            .execute_proof(
                merk_proof_bytes,
                layer_limit,
                left_to_right,
                PROOF_VERSION_LATEST, // V1 proof: strict mode rejects items in value hash nodes
            )
            .unwrap()
            .map_err(|e| {
                invalid_proof(
                    path_query,
                    format!("Invalid V1 proof verification parameters: {}", e),
                )
            })?;

        let mut verified_keys = BTreeSet::new();

        if merk_result.result_set.is_empty() {
            charge_empty_v1_layer(
                limit_state,
                prove_options.decrease_limit_on_empty_sub_query_result,
            );
        } else {
            let ctx = V1LayerCtx {
                lower_layers,
                prove_options,
                query: path_query,
                internal_query: &internal_query,
                current_path,
                options,
                current_depth,
                grove_version,
            };
            let rows = merk_result.result_set;
            Self::verify_v1_rows(
                ctx,
                &rows,
                0,
                rows.len(),
                &mut verified_keys,
                limit_state,
                &mut frame_instance,
                result,
                axis_outcomes,
                last_parent_tree_type,
            )?;
        }

        // Succinctness check: reject proof if it contains lower layers
        // for keys that the query did not require
        if options.verify_proof_succinctness
            && let Some(key) = first_unrequested_lower_layer(lower_layers, &verified_keys)
        {
            return Err(invalid_proof(
                path_query,
                format!(
                    "V1 proof contains extra lower layer for key {} not required by query",
                    hex::encode(key),
                ),
            ));
        }

        Ok(root_hash)
    }

    /// Verify `rows[lo..hi]` in order, stopping at the first row that
    /// returns `RowFlow::Break` (which is returned) or an error: the row
    /// loop of `verify_layer_proof_v1`.
    ///
    /// The range is halved, the left half verified, and the right half
    /// verified only if the left one did not stop, so the rows are visited
    /// in exactly the order and with exactly the early exits of
    /// `for row in rows { match verify_v1_row(row)? { Break => break, .. } }`.
    /// Recursion rather than a loop; halving keeps the depth at
    /// `log2(hi - lo) + 1`. Those frames stay live while a row descends to
    /// its lower layer, so a proof `d` levels deep adds up to `d` times that
    /// to the stack (`d <= MAX_PROOF_DEPTH`).
    #[allow(clippy::too_many_arguments)]
    fn verify_v1_rows<S: RowSink>(
        ctx: V1LayerCtx<'_>,
        rows: &[ProvedKeyOptionalValue],
        lo: usize,
        hi: usize,
        verified_keys: &mut BTreeSet<Vec<u8>>,
        limit_state: &mut super::V1LimitState,
        frame_instance: &mut Option<u16>,
        result: &mut S,
        axis_outcomes: &mut Vec<AxisWalkOutcome>,
        last_parent_tree_type: &mut Option<TreeFeatureType>,
    ) -> Result<RowFlow, Error> {
        if hi <= lo {
            Ok(RowFlow::Continue)
        } else if hi - lo == 1 {
            Self::verify_v1_row(
                ctx,
                &rows[lo],
                verified_keys,
                limit_state,
                frame_instance,
                result,
                axis_outcomes,
                last_parent_tree_type,
            )
        } else {
            let mid = lo + (hi - lo) / 2;
            match Self::verify_v1_rows(
                ctx,
                rows,
                lo,
                mid,
                verified_keys,
                limit_state,
                frame_instance,
                result,
                axis_outcomes,
                last_parent_tree_type,
            )? {
                RowFlow::Break => Ok(RowFlow::Break),
                RowFlow::Continue => Self::verify_v1_rows(
                    ctx,
                    rows,
                    mid,
                    hi,
                    verified_keys,
                    limit_state,
                    frame_instance,
                    result,
                    axis_outcomes,
                    last_parent_tree_type,
                ),
            }
        }
    }

    /// One row of a V1 layer: the body of the row loop in
    /// `verify_layer_proof_v1`. `RowFlow::Break` is the loop's `break`
    /// (a limit ran out); every other completed row is
    /// `RowFlow::Continue`.
    #[allow(clippy::too_many_arguments)]
    fn verify_v1_row<S: RowSink>(
        ctx: V1LayerCtx<'_>,
        proved_key_value: &ProvedKeyOptionalValue,
        verified_keys: &mut BTreeSet<Vec<u8>>,
        limit_state: &mut super::V1LimitState,
        frame_instance: &mut Option<u16>,
        result: &mut S,
        axis_outcomes: &mut Vec<AxisWalkOutcome>,
        last_parent_tree_type: &mut Option<TreeFeatureType>,
    ) -> Result<RowFlow, Error> {
        let V1LayerCtx {
            lower_layers,
            query,
            internal_query,
            current_path,
            options,
            grove_version,
            ..
        } = ctx;
        let mut path = current_path.to_vec();
        let key = &proved_key_value.key;
        let hash = &proved_key_value.proof;
        if let Some(value_bytes) = &proved_key_value.value {
            // Look through NonCounted: dispatch on inner element. The
            // serialized value bytes used for hashing keep their
            // wrapper byte, so cryptographic verification still works.
            let element = Element::deserialize(value_bytes, grove_version)?.into_underlying();

            // A raw reference can never be an honest V1 row. The
            // prover rewrites every reference the verifier will
            // consume into a `KVRefValueHash*` node carrying the
            // dereferenced TARGET bytes, so the bytes surfaced
            // here are the target's — never the reference's own.
            // A `KVValueHash` / `KVValueHashFeatureType` node
            // hashes only `(key, value_hash)`, and the merk-level
            // V1 guard refuses only *items* on those forms (it
            // cannot refuse references: a reference row past the
            // query limit legitimately stays bare there, and this
            // loop stops before reading it). So a forged proof can
            // pair genuine `value_hash` with arbitrary reference
            // bytes and still reconstruct the root. This is the
            // check that binds the row: refuse before the element
            // type can be returned as unauthenticated reference
            // metadata, or make "a reference has no lower layer"
            // silently skip a descent the query asked for
            // (issue #862).
            if element.is_reference() {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 proof surfaces a raw {} at key {}: the prover serves \
                         references as KVRefValueHash nodes carrying the dereferenced \
                         target, so an unresolved reference row is unbound and \
                         malformed",
                        element.type_str(),
                        hex::encode(key),
                    ),
                ));
            }

            verified_keys.insert(key.clone());

            // Axis-ordered read: resolved from the QUERY first —
            // before the element type is consulted and before
            // the lower-layer lookup — so that neither a
            // wrong-typed element nor a missing layer can
            // fabricate an absence. If the query node governing
            // this key carries ReadMode::Axis then this key is
            // an axis read, and the only proof that answers it
            // is an axis descent over an indexed tree:
            //
            //   * a non-indexed element here is a hard error. It
            //     used to fall through to the ordinary descent,
            //     which yields no trio for a present normal tree
            //     — so the branched shape saw neither an axis
            //     layer nor a present element and endorsed the
            //     branch as `None`, i.e. proven absent, while the
            //     trusted read rejects the same state as
            //     unindexed.
            //   * a missing layer is a hard error, not a silent
            //     skip (which the branched shape would likewise
            //     endorse as a proven-absent slot).
            //
            // An empty indexed tree still reports an EMPTY axis
            // result (verify_axis_descent_layer proves it against
            // a NULL-hash secondary), matching the honest prover
            // and the trusted read rather than reporting the
            // branch absent.
            let mut axis_path = current_path.to_vec();
            axis_path.push(key);
            if let Some(axis_query) = query.axis_read_at_path(&axis_path) {
                if !element.is_indexed_tree() {
                    return Err(invalid_proof(
                        query,
                        format!(
                            "V1 axis read at key {} resolves to a non-indexed \
                             element ({}); an axis-ordered read has no meaning \
                             there, so the proof cannot answer this query — and \
                             the key must not be reported as absent",
                            hex::encode(key),
                            element.type_str(),
                        ),
                    ));
                }
                let Some(lower_layer) = lower_layers.get(key) else {
                    return Err(invalid_proof(
                        query,
                        format!(
                            "V1 axis read at key {} has no axis-descent lower \
                             layer; the element cannot be reported as absent",
                            hex::encode(key),
                        ),
                    ));
                };
                path.push(key);
                *last_parent_tree_type = element.tree_feature_type();
                Self::verify_axis_descent_layer(
                    lower_layer,
                    axis_query,
                    value_bytes,
                    hash,
                    &path,
                    axis_outcomes,
                    query,
                    grove_version,
                )?;
                return Ok(RowFlow::Continue);
            }

            if let Some(lower_layer) = lower_layers.get(key) {
                // MmrTree/BulkAppendTree have root_key=None (no child Merk data),
                // so they match on (..) rather than (Some(_), ..)
                match element {
                    // Indexed-tree descent (PCIT / PSIT / PCPSIT):
                    // descend into the primary as if it were a
                    // standard Merk subtree, then chain via
                    // combine_hash_three(value_hash, primary_root,
                    // attestation) where `attestation` is the 32-byte
                    // prefix on the ProofBytes::CountIndexedTree
                    // envelope. For PCIT/PSIT the prefix is the single
                    // secondary Merk root hash; for PCPSIT it is the
                    // axes_digest over every axis's secondary root
                    // hash. The verifier treats the prefix opaquely as
                    // the third combine input, so the same descent
                    // logic serves all three variants — matching the
                    // shared prover envelope in generate.rs.
                    Element::ProvableCountIndexedTree(..)
                    | Element::ProvableSumIndexedTree(..)
                    | Element::ProvableCountProvableSumIndexedTree(..) => {
                        return Self::verify_indexed_lower_layer(
                            ctx,
                            proved_key_value,
                            &element,
                            value_bytes,
                            lower_layer,
                            path,
                            limit_state,
                            frame_instance,
                            result,
                            axis_outcomes,
                            last_parent_tree_type,
                        );
                    }
                    // The V1 prover rejects subqueries into a
                    // PrivateDocumentStore (range-read proofs are not
                    // implemented yet), so an honest proof never
                    // carries a lower layer for one — refuse to
                    // fabricate a chain for it.
                    Element::PrivateDocumentStore(..) => {
                        return Err(Error::NotSupported(
                            "V1 proofs do not yet support lower layers for \
                             PrivateDocumentStore elements"
                                .to_string(),
                        ));
                    }
                    Element::Tree(Some(_), _)
                    | Element::SumTree(Some(_), ..)
                    | Element::BigSumTree(Some(_), ..)
                    | Element::CountTree(Some(_), ..)
                    | Element::CountSumTree(Some(_), ..)
                    | Element::ProvableCountTree(Some(_), ..)
                    | Element::ProvableCountSumTree(Some(_), ..)
                    | Element::ProvableSumTree(Some(_), ..)
                    | Element::ProvableCountProvableSumTree(Some(_), ..)
                    | Element::CommitmentTree(..)
                    | Element::MmrTree(..)
                    | Element::BulkAppendTree(..)
                    | Element::DenseAppendOnlyFixedSizeTree(..) => {
                        return Self::verify_tree_lower_layer(
                            ctx,
                            proved_key_value,
                            &element,
                            value_bytes,
                            lower_layer,
                            path,
                            limit_state,
                            frame_instance,
                            result,
                            axis_outcomes,
                            last_parent_tree_type,
                        );
                    }
                    // MmrTree/BulkAppendTree/CommitmentTree handled above
                    Element::Tree(None, _)
                    | Element::SumTree(None, ..)
                    | Element::BigSumTree(None, ..)
                    | Element::CountTree(None, ..)
                    | Element::CountSumTree(None, ..)
                    | Element::ProvableCountTree(None, ..)
                    | Element::ProvableCountSumTree(None, ..)
                    | Element::ProvableSumTree(None, ..)
                    | Element::ProvableCountProvableSumTree(None, ..)
                    | Element::SumItem(..)
                    | Element::Item(..)
                    | Element::ItemWithSumItem(..)
                    | Element::Reference(..)
                    | Element::ReferenceWithSumItem(..)
                    | Element::BidirectionalReference(..)
                    | Element::ItemWithBackwardsReferences(..)
                    | Element::SumItemWithBackwardsReferences(..)
                    | Element::ItemWithSumItemWithBackwardsReferences(..) => {
                        return Err(invalid_proof(
                            query,
                            "V1 proof has lower layer for a non-tree element.".to_string(),
                        ));
                    }
                    // PSIT / PCPSIT descent is handled together with
                    // PCIT in the shared indexed-tree arm above.
                    Element::NonCounted(_)
                    | Element::NotSummed(_)
                    | Element::NotCountedOrSummed(_) => {
                        unreachable!("unwrapped above")
                    }
                }
            } else if is_terminal_row(&element, internal_query, key, options) {
                return Self::verify_terminal_row(
                    ctx,
                    proved_key_value,
                    &element,
                    value_bytes,
                    path,
                    limit_state,
                    frame_instance,
                    result,
                );
            } else if element.is_any_tree()
                && !element.is_non_empty_tree()
                && internal_query.has_subquery_or_matching_in_path_on_key(key)
            {
                return Self::verify_empty_subquery_tree(
                    ctx,
                    proved_key_value,
                    &element,
                    value_bytes,
                    path,
                    limit_state,
                    frame_instance,
                    result,
                    last_parent_tree_type,
                );
            } else if element.is_non_empty_tree()
                && internal_query.has_subquery_or_matching_in_path_on_key(key)
            {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 proof is missing lower layer for non-empty tree at key {}",
                        hex::encode(key),
                    ),
                ));
            } else if element.is_any_tree() && !element.is_non_empty_tree() {
                // A claimed-empty plain `Tree` at a terminal
                // position while the caller excludes empty trees
                // from results (`include_empty_trees_in_result`
                // is false). It is not reported, but the claim
                // still decides that a key the trusted read
                // returns is absent from the result set — so it
                // must be bound exactly like a reported empty
                // tree (issue #869).
                Self::verify_empty_tree_binding(query, key, &element, value_bytes, hash)?;
            } else {
                // Every classification the verifier can act on
                // is either descended into (bound through its
                // lower layer), reported (bound above), or a
                // bound empty tree. Nothing else may be
                // silently skipped: the element bytes of a
                // `KVValueHash*` node are unauthenticated until
                // one of those paths binds them.
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 proof reports an element at key {} that is neither \
                         descended into nor bound to the proof",
                        hex::encode(key),
                    ),
                ));
            }
        }
        Ok(RowFlow::Continue)
    }

    /// A V1 row whose element is an indexed tree (PCIT / PSIT / PCPSIT)
    /// with a lower layer: the indexed-tree arm of `verify_v1_row`.
    #[allow(clippy::too_many_arguments)]
    fn verify_indexed_lower_layer<'a, S: RowSink>(
        ctx: V1LayerCtx<'a>,
        proved_key_value: &'a ProvedKeyOptionalValue,
        element: &Element,
        value_bytes: &'a [u8],
        lower_layer: &'a LayerProof,
        mut path: Vec<&'a [u8]>,
        limit_state: &mut super::V1LimitState,
        frame_instance: &mut Option<u16>,
        result: &mut S,
        axis_outcomes: &mut Vec<AxisWalkOutcome>,
        last_parent_tree_type: &mut Option<TreeFeatureType>,
    ) -> Result<RowFlow, Error> {
        let V1LayerCtx {
            prove_options,
            query,
            current_path,
            options,
            current_depth,
            grove_version,
            ..
        } = ctx;
        let key = &proved_key_value.key;
        let hash = &proved_key_value.proof;
        // An axis-read indexed tree was already
        // handled (and `continue`d) above, before
        // this lower-layer lookup, so here the
        // query descends the primary as usual.
        path.push(key);
        *last_parent_tree_type = element.tree_feature_type();
        if query.query_items_at_path(&path, grove_version)?.is_none() {
            // The indexed tree is itself a result and
            // nothing is queried below it. The element
            // bytes still have to be bound to the
            // parent-committed `value_hash`: a
            // `KVValueHash` node hashes only
            // (key, value_hash), so pushing the value
            // unchecked would let a prover swap in a
            // forged element (any count/sum it likes)
            // under a genuine root hash. Only the
            // terminal envelope can bind it here —
            // a descent envelope is unusable because
            // there is no query at the lower path to
            // verify it against.
            let ProofBytes::IndexedTreeTerminal(terminal_bytes) = &lower_layer.merk_proof else {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 proof supplies a non-terminal lower layer for \
                         indexed tree at key {} with no query below it; \
                         the element bytes would be unbound",
                        hex::encode(key),
                    ),
                ));
            };
            if !lower_layer.lower_layers.is_empty() {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 indexed terminal attestation at key {} must \
                         not carry further lower layers",
                        hex::encode(key),
                    ),
                ));
            }
            if terminal_bytes.len() != 64 {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 indexed terminal attestation at key {} must \
                         be exactly 64 bytes (attestation || \
                         primary_root), got {}",
                        hex::encode(key),
                        terminal_bytes.len(),
                    ),
                ));
            }
            let mut attestation = [0u8; 32];
            attestation.copy_from_slice(&terminal_bytes[..32]);
            let mut primary_root = [0u8; 32];
            primary_root.copy_from_slice(&terminal_bytes[32..]);

            let combined_root_hash =
                combine_hash_three(value_hash(value_bytes).value(), &primary_root, &attestation)
                    .value()
                    .to_owned();
            if hash != &combined_root_hash {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 mismatch in indexed terminal attestation at \
                         key {}, expected {}, got {}",
                        hex::encode(key),
                        hex::encode(hash),
                        hex::encode(combined_root_hash)
                    ),
                ));
            }

            // Report under the PARENT path (path
            // without the trailing key we pushed for
            // descent), matching query_raw and the
            // regular-tree terminal arm. Including the
            // key made `(path, key)` lookups miss —
            // absence-proof verification would report
            // an EXISTING indexed tree as absent.
            let parent_path: Vec<Vec<u8>> = path_to_owned(&path[..path.len().saturating_sub(1)]);
            let path_key_optional_value = ProvedPathKeyOptionalValue::from_proved_key_value(
                parent_path,
                proved_key_value.clone(),
            );
            result.push_row(path_key_optional_value, grove_version)?;
            limit_state.charge_row_with_instance(frame_instance);
            if limit_state.is_exhausted(*frame_instance) {
                return Ok(RowFlow::Break);
            }
        } else {
            if query.should_add_parent_tree_at_path(current_path, grove_version)? {
                let path_key_optional_value = ProvedPathKeyOptionalValue::from_proved_key_value(
                    path_to_owned(&path),
                    proved_key_value.clone(),
                );
                result.push_row(path_key_optional_value, grove_version)?;
            }

            // Lower layer must carry cidx-shaped
            // proof bytes: secondary_root_hash (32)
            // followed by a standard Merk proof
            // against the primary.
            let cidx_bytes = match &lower_layer.merk_proof {
                ProofBytes::CountIndexedTree(b) => b,
                _ => {
                    return Err(invalid_proof(
                        query,
                        "V1 lower layer for CountIndexedTree element \
                         must use ProofBytes::CountIndexedTree"
                            .to_string(),
                    ));
                }
            };
            if cidx_bytes.len() < 32 {
                return Err(invalid_proof(
                    query,
                    "cidx proof bytes shorter than 32-byte secondary \
                     root attestation prefix"
                        .to_string(),
                ));
            }
            let mut secondary_root: [u8; 32] = [0u8; 32];
            secondary_root.copy_from_slice(&cidx_bytes[..32]);
            // Recurse straight into the primary proof
            // carved out of the indexed envelope,
            // borrowing this layer's descendants. This
            // used to synthesize a LayerProof, which
            // deep-copied `lower_layers` at every
            // nesting level — memory an untrusted peer
            // could multiply by nesting depth.
            let rows_before_descent = limit_state.consumed_rows;
            let primary_root_hash = Self::verify_layer_proof_v1(
                &cidx_bytes[32..],
                &lower_layer.lower_layers,
                prove_options,
                query,
                limit_state,
                *frame_instance,
                &path,
                result,
                axis_outcomes,
                last_parent_tree_type,
                options,
                current_depth + 1,
                grove_version,
            )?;
            super::V1LimitState::settle_instance_after_descent(
                frame_instance,
                rows_before_descent,
                limit_state.consumed_rows,
            );

            let combined_root_hash = combine_hash_three(
                value_hash(value_bytes).value(),
                &primary_root_hash,
                &secondary_root,
            )
            .value()
            .to_owned();

            if hash != &combined_root_hash {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 mismatch in cidx lower-layer hash, \
                         expected {}, got {}",
                        hex::encode(hash),
                        hex::encode(combined_root_hash)
                    ),
                ));
            }
            if limit_state.is_exhausted(*frame_instance) {
                return Ok(RowFlow::Break);
            }
        }
        Ok(RowFlow::Continue)
    }

    /// A V1 row whose element is a non-empty Merk tree or a non-Merk tree
    /// with a lower layer: the tree arm of `verify_v1_row`.
    #[allow(clippy::too_many_arguments)]
    fn verify_tree_lower_layer<'a, S: RowSink>(
        ctx: V1LayerCtx<'a>,
        proved_key_value: &'a ProvedKeyOptionalValue,
        element: &Element,
        value_bytes: &'a [u8],
        lower_layer: &'a LayerProof,
        mut path: Vec<&'a [u8]>,
        limit_state: &mut super::V1LimitState,
        frame_instance: &mut Option<u16>,
        result: &mut S,
        axis_outcomes: &mut Vec<AxisWalkOutcome>,
        last_parent_tree_type: &mut Option<TreeFeatureType>,
    ) -> Result<RowFlow, Error> {
        let V1LayerCtx {
            prove_options,
            query,
            current_path,
            options,
            current_depth,
            grove_version,
            ..
        } = ctx;
        let key = &proved_key_value.key;
        let hash = &proved_key_value.proof;
        path.push(key);
        *last_parent_tree_type = element.tree_feature_type();
        // Does the current query ask for anything BELOW
        // this tree, or only for the tree element
        // itself?
        //
        // "Only the element itself" while the proof
        // still carries a lower layer is the ordinary
        // shape of a SUBSET verification: the proof was
        // generated for a wider query that descended
        // here, and is now being re-verified against a
        // narrower one that stops at the tree. Dash
        // Platform does exactly this to read a
        // CommitmentTree's total note count out of the
        // note-fetch proof it already has.
        //
        // Either way the element bytes must stay bound
        // to the parent-committed `value_hash`: a
        // `KVValueHash`-family node hashes only
        // (key, value_hash), so reporting `value`
        // unchecked would let a prover attach a dummy
        // lower layer and substitute forged element
        // bytes under a genuine root hash. The
        // `combine_hash` chain check below is what binds
        // it, and it needs the lower layer's root — so
        // the lower layer is consumed for its root hash
        // in BOTH cases. Only the reporting differs:
        // with no query below, none of the lower layer's
        // rows belong in this query's result set, so the
        // tree element itself is reported instead.
        let has_query_below = query.query_items_at_path(&path, grove_version)?.is_some();

        if !has_query_below && options.verify_proof_succinctness {
            // Succinct mode demands the proof carry
            // nothing beyond what the query needs, and
            // an honest prover never emits this layer
            // for this query (every `lower_layers.insert`
            // site is gated on there being a subquery
            // for the key). Reject as extra data.
            return Err(invalid_proof(
                query,
                format!(
                    "V1 proof supplies a lower layer for tree at key {} \
                     with no query below it; the element bytes would be \
                     unbound",
                    hex::encode(key),
                ),
            ));
        }

        {
            // Known limitation: this parent tree result
            // is pushed without decrementing limit_left.
            // Will be addressed by per-level limits
            // redesign.
            if has_query_below
                && query.should_add_parent_tree_at_path(current_path, grove_version)?
            {
                let path_key_optional_value = ProvedPathKeyOptionalValue::from_proved_key_value(
                    path_to_owned(&path),
                    proved_key_value.clone(),
                );
                result.push_row(path_key_optional_value, grove_version)?;
            }

            // Dispatch based on lower layer proof type.
            // `has_query_below == false` derives the
            // layer's root WITHOUT reporting any of its
            // contents — every lower-layer flavour
            // computes its root independently of the
            // query, which only ever selects rows.
            let rows_before_descent = limit_state.consumed_rows;
            let parent = ParentCommitment {
                value_bytes,
                value_hash: hash,
            };
            // Every arm binds the root it derives to
            // `parent`; the append-only ones do it
            // before reporting any row.
            match &lower_layer.merk_proof {
                ProofBytes::Merk(_) => {
                    // A sum-budget layer must carry
                    // the window envelope; a plain
                    // Merk descent here would let a
                    // prover serve a read-mode query
                    // as key selection.
                    if query.sum_budget_read_at_path(&path).is_some() {
                        return Err(invalid_proof(
                            query,
                            "the query node at this tree is a \
                             sum-budget read; its lower layer must \
                             carry ProofBytes::SumBudgetWindow"
                                .to_string(),
                        ));
                    }
                    // Standard Merk subtree - recurse
                    let merk_bytes = Self::merk_bytes_of_layer(lower_layer, query)?;
                    let lower_hash = if has_query_below {
                        Self::verify_layer_proof_v1(
                            merk_bytes,
                            &lower_layer.lower_layers,
                            prove_options,
                            query,
                            limit_state,
                            *frame_instance,
                            &path,
                            result,
                            axis_outcomes,
                            last_parent_tree_type,
                            options,
                            current_depth + 1,
                            grove_version,
                        )?
                    } else {
                        Self::merk_layer_root_hash(merk_bytes, query)?
                    };
                    parent.bind(query, &lower_hash)?;
                }
                ProofBytes::SumBudgetWindow(payload_bytes) => {
                    if !lower_layer.lower_layers.is_empty() {
                        return Err(invalid_proof(
                            query,
                            "a sum-budget window is terminal and must \
                             not carry further lower layers"
                                .to_string(),
                        ));
                    }
                    let lower_hash = Self::verify_sum_budget_window_layer(
                        payload_bytes,
                        &path,
                        axis_outcomes,
                        query,
                        grove_version,
                    )?;
                    parent.bind(query, &lower_hash)?;
                }
                ProofBytes::MMR(mmr_bytes) => {
                    // Mirror of the prover: min-compose
                    // the lower query's own per-instance
                    // cap — these adapters bypass the
                    // recursive frame creation.
                    let mut non_merk_effective = limit_state.effective_lower_layer_limit(
                        *frame_instance,
                        query
                            .query_items_at_path(&path, grove_version)?
                            .and_then(|lower_query| lower_query.instance_limit),
                    );
                    let non_merk_before = non_merk_effective;
                    Self::verify_mmr_lower_layer(
                        mmr_bytes,
                        element,
                        parent,
                        &path,
                        &mut non_merk_effective,
                        result,
                        query,
                        has_query_below,
                        grove_version,
                    )?;
                    limit_state.charge_rows(
                        non_merk_before
                            .unwrap_or(0)
                            .saturating_sub(non_merk_effective.unwrap_or(0)),
                    );
                }
                ProofBytes::BulkAppendTree(bulk_bytes) => {
                    // Mirror of the prover: min-compose
                    // the lower query's own per-instance
                    // cap — these adapters bypass the
                    // recursive frame creation.
                    let mut non_merk_effective = limit_state.effective_lower_layer_limit(
                        *frame_instance,
                        query
                            .query_items_at_path(&path, grove_version)?
                            .and_then(|lower_query| lower_query.instance_limit),
                    );
                    let non_merk_before = non_merk_effective;
                    Self::verify_bulk_append_lower_layer(
                        bulk_bytes,
                        element,
                        parent,
                        &path,
                        &mut non_merk_effective,
                        result,
                        query,
                        has_query_below,
                        grove_version,
                    )?;
                    limit_state.charge_rows(
                        non_merk_before
                            .unwrap_or(0)
                            .saturating_sub(non_merk_effective.unwrap_or(0)),
                    );
                }
                ProofBytes::DenseTree(dense_bytes) => {
                    // Mirror of the prover: min-compose
                    // the lower query's own per-instance
                    // cap — these adapters bypass the
                    // recursive frame creation.
                    let mut non_merk_effective = limit_state.effective_lower_layer_limit(
                        *frame_instance,
                        query
                            .query_items_at_path(&path, grove_version)?
                            .and_then(|lower_query| lower_query.instance_limit),
                    );
                    let non_merk_before = non_merk_effective;
                    Self::verify_dense_tree_lower_layer(
                        dense_bytes,
                        element,
                        parent,
                        &path,
                        &mut non_merk_effective,
                        result,
                        query,
                        has_query_below,
                        grove_version,
                    )?;
                    limit_state.charge_rows(
                        non_merk_before
                            .unwrap_or(0)
                            .saturating_sub(non_merk_effective.unwrap_or(0)),
                    );
                }
                ProofBytes::CommitmentTree(ct_bytes) => {
                    // Mirror of the prover: min-compose
                    // the lower query's own per-instance
                    // cap — these adapters bypass the
                    // recursive frame creation.
                    let mut non_merk_effective = limit_state.effective_lower_layer_limit(
                        *frame_instance,
                        query
                            .query_items_at_path(&path, grove_version)?
                            .and_then(|lower_query| lower_query.instance_limit),
                    );
                    let non_merk_before = non_merk_effective;
                    Self::verify_commitment_tree_lower_layer(
                        ct_bytes,
                        element,
                        parent,
                        &path,
                        &mut non_merk_effective,
                        result,
                        query,
                        has_query_below,
                        grove_version,
                    )?;
                    limit_state.charge_rows(
                        non_merk_before
                            .unwrap_or(0)
                            .saturating_sub(non_merk_effective.unwrap_or(0)),
                    );
                }
                ProofBytes::CountIndexedTree(_)
                | ProofBytes::IndexedTreeTerminal(_)
                | ProofBytes::IndexedTreeAxisDescent(_) => {
                    // Indexed-tree envelopes are
                    // dispatched in the dedicated
                    // arm above; reaching this point
                    // means the proof envelope put
                    // a cidx-shaped lower layer on
                    // a non-cidx parent element,
                    // which is malformed.
                    return Err(invalid_proof(
                        query,
                        "indexed-tree proof envelope under a non-indexed \
                         parent element"
                            .to_string(),
                    ));
                }
            };

            super::V1LimitState::settle_instance_after_descent(
                frame_instance,
                rows_before_descent,
                limit_state.consumed_rows,
            );

            if !has_query_below {
                // The tree element itself is the
                // result, now bound by the
                // `combine_hash` check above. Report it
                // under the PARENT path (not the path
                // we pushed the key onto for the root
                // derivation), matching query_raw and
                // the no-lower-layer terminal arm —
                // including the key would make
                // `(path, key)` lookups miss.
                let parent_path: Vec<Vec<u8>> = path_to_owned(current_path);
                let path_key_optional_value = ProvedPathKeyOptionalValue::from_proved_key_value(
                    parent_path,
                    proved_key_value.clone(),
                );
                result.push_row(path_key_optional_value, grove_version)?;
                limit_state.charge_row_with_instance(frame_instance);
            }

            if limit_state.is_exhausted(*frame_instance) {
                return Ok(RowFlow::Break);
            }
        }
        Ok(RowFlow::Continue)
    }

    /// A V1 row without a lower layer that is reported as a result: an
    /// item, or a tree the query does not descend into.
    #[allow(clippy::too_many_arguments)]
    fn verify_terminal_row<'a, S: RowSink>(
        ctx: V1LayerCtx<'a>,
        proved_key_value: &'a ProvedKeyOptionalValue,
        element: &Element,
        value_bytes: &'a [u8],
        path: Vec<&'a [u8]>,
        limit_state: &mut super::V1LimitState,
        frame_instance: &mut Option<u16>,
        result: &mut S,
    ) -> Result<RowFlow, Error> {
        let V1LayerCtx {
            query,
            grove_version,
            ..
        } = ctx;
        let key = &proved_key_value.key;
        let hash = &proved_key_value.proof;
        // For empty trees in the result set (no lower layer
        // proof), bind the element bytes to the proof's
        // value_hash before trusting anything decoded from
        // them (see `verify_empty_tree_binding`).
        Self::verify_empty_tree_binding(query, key, element, value_bytes, hash)?;

        // For trees reported without a subquery (no lower layer
        // proof), the prover must use
        // KVValueHashFeatureTypeWithChildHash so the merk
        // verifier can confirm combine_hash(H(value),
        // child_hash) == value_hash. If child_hash_verified is
        // false, an attacker may have downgraded the node type
        // to hide child hash verification.
        //
        // Non-empty Merk trees (child_hash = child Merk root)
        // have required this since V3 and are checked at every
        // version.
        //
        // The four non-Merk trees — CommitmentTree, MmrTree,
        // BulkAppendTree, DenseAppendOnlyFixedSizeTree
        // (child_hash = the tree's own state root, which their
        // parent commits through the same two-input
        // combine_hash) — are checked only from V4, under
        // `proof.terminal_non_merk_tree_child_hash`. V1..V3
        // provers emit a bare KVValueHash here, so demanding
        // the child hash from them would reject honest
        // released proofs; the gate moves prover and verifier
        // together at the protocol boundary. Until it
        // activates, the element bytes of a terminally-reported
        // non-Merk tree stay unbound and a prover can forge the
        // entry count callers read from them.
        //
        // `is_non_empty_tree` is true unconditionally for those
        // four, so an empty one is bound too — its committed
        // child hash is NULL_HASH, or
        // EMPTY_COMMITMENT_TREE_STATE_ROOT for a
        // CommitmentTree.
        let requires_child_hash = terminal_requires_child_hash(element, grove_version);
        if requires_child_hash && !proved_key_value.child_hash_verified {
            return Err(invalid_proof(
                query,
                format!(
                    "V1 non-empty tree at key {} without subquery must use \
                     KVValueHashFeatureTypeWithChildHash proof node",
                    hex::encode(key),
                ),
            ));
        }

        let path_key_optional_value = ProvedPathKeyOptionalValue::from_proved_key_value(
            path_to_owned(&path),
            proved_key_value.clone(),
        );
        result.push_row(path_key_optional_value, grove_version)?;
        limit_state.charge_row_with_instance(frame_instance);
        if limit_state.is_exhausted(*frame_instance) {
            return Ok(RowFlow::Break);
        }
        Ok(RowFlow::Continue)
    }

    /// A V1 row whose element is an empty tree a subquery matches, so
    /// there is no lower layer to descend into.
    #[allow(clippy::too_many_arguments)]
    fn verify_empty_subquery_tree<'a, S: RowSink>(
        ctx: V1LayerCtx<'a>,
        proved_key_value: &'a ProvedKeyOptionalValue,
        element: &Element,
        value_bytes: &'a [u8],
        path: Vec<&'a [u8]>,
        limit_state: &mut super::V1LimitState,
        frame_instance: &mut Option<u16>,
        result: &mut S,
        last_parent_tree_type: &mut Option<TreeFeatureType>,
    ) -> Result<RowFlow, Error> {
        let V1LayerCtx {
            query,
            current_path,
            grove_version,
            ..
        } = ctx;
        let key = &proved_key_value.key;
        let hash = &proved_key_value.proof;
        // An EMPTY tree a subquery matches: an empty
        // child, not a result row. "Empty" is so far
        // only what the element bytes CLAIM, and a
        // `KVValueHash*` node does not hash those bytes
        // — a prover can rewrite a populated tree into
        // its empty form, omit the lower layer, and the
        // root still verifies. Bind the claim to the
        // proof's value_hash before acting on it; the
        // real tree's child hash is NULL_HASH only if it
        // really is empty (issue #869).
        Self::verify_empty_tree_binding(query, key, element, value_bytes, hash)?;
        // There is no lower layer to descend into, but this
        // verified empty tree still supplies the parent-info
        // result. Do not leave a populated ancestor's metadata
        // behind (or None when the target is a root child).
        *last_parent_tree_type = element.tree_feature_type();
        // When the governing query asks for parent-tree
        // inclusion, the matched parent is still
        // reported — empty and non-empty parents must
        // behave alike — and, like every parent-tree
        // row, it does not consume a budget slot (the
        // documented known limitation on the flag).
        if query.should_add_parent_tree_at_path(current_path, grove_version)? {
            let path_key_optional_value = ProvedPathKeyOptionalValue::from_proved_key_value(
                path_to_owned(&path),
                proved_key_value.clone(),
            );
            result.push_row(path_key_optional_value, grove_version)?;
        }
        // Mirror the prover's charge exactly — global
        // budget only (it bounds walks across many
        // empty children), per-instance budgets count
        // result rows.
        limit_state.charge_empty_layer();
        if limit_state.is_exhausted(*frame_instance) {
            return Ok(RowFlow::Break);
        }
        Ok(RowFlow::Continue)
    }

    /// Bind a claimed-empty tree's element bytes to the proof's
    /// authenticated `value_hash`.
    ///
    /// A merk `KVValueHash*` node hashes `(key, value_hash)`; the element
    /// bytes it carries are NOT part of the merk root, so everything the
    /// verifier decodes from them — tree type, emptiness — is a claim
    /// until it is tied to `value_hash`. A non-empty tree is tied through
    /// the lower layer it descends into (`combine_hash(H(value), lower
    /// root)`) or the child hash it must carry when reported terminally.
    /// An empty tree has no lower layer, so the tie is that its committed
    /// child hash is NULL_HASH: `value_hash == combine_hash(H(value),
    /// NULL_HASH)`. Without this a prover could swap tree types
    /// (SumTree→Tree) or rewrite a populated tree into its empty form —
    /// omitting the descent — without breaking the merk proof.
    ///
    /// Indexed-tree elements (PCIT / PSIT / PCPSIT) have TWO (or more)
    /// child Merks — a primary and one-or-more secondaries — so their
    /// empty form commits `combine_hash_three(H(value), NULL_HASH,
    /// second)` rather than the two-input `combine_hash`:
    ///   - Empty PCIT / PSIT: `second = NULL_HASH` (the lone secondary's
    ///     root hash while empty).
    ///   - Empty PCPSIT: `second = axes_digest(zero_axes)`, the digest
    ///     over the element's own axes list with every axis's secondary
    ///     root hash = NULL_HASH.
    ///
    /// These mirror the insert commit path in
    /// add_element_on_transaction/v1.rs.
    ///
    /// GROVE_V1/V2 direct inserts committed empty `CountSumTree`,
    /// `ProvableCountTree`, and `ProvableCountSumTree` as `H(value)`.
    /// Those rows can coexist with layered commitments after an upgrade,
    /// so accept that form for these types only when it authenticates the
    /// exact serialized bytes, independently of the reader's version.
    ///
    /// Non-tree elements and non-empty trees pass through untouched:
    /// their binding is the caller's responsibility.
    fn verify_empty_tree_binding(
        query: &PathQuery,
        key: &[u8],
        element: &Element,
        value_bytes: &[u8],
        hash: &CryptoHash,
    ) -> Result<(), Error> {
        let empty_indexed_expected: Option<CryptoHash> = match element {
            Element::ProvableCountIndexedTree(None, None, 0, _)
            | Element::ProvableSumIndexedTree(None, None, 0, _) => Some(
                combine_hash_three(value_hash(value_bytes).value(), &NULL_HASH, &NULL_HASH)
                    .value()
                    .to_owned(),
            ),
            Element::ProvableCountProvableSumIndexedTree(None, 0, 0, axes, _)
                if all_axes_unset(axes) =>
            {
                let zero_axes: Vec<(u8, CryptoHash)> = null_axis_hashes(axes);
                let digest = axes_digest(&zero_axes).value().to_owned();
                Some(
                    combine_hash_three(value_hash(value_bytes).value(), &NULL_HASH, &digest)
                        .value()
                        .to_owned(),
                )
            }
            _ => None,
        };
        if let Some(expected_value_hash) = empty_indexed_expected {
            if hash != &expected_value_hash {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 empty indexed-tree value hash mismatch at key {}: expected {}, got {}",
                        hex::encode(key),
                        hex::encode(hash),
                        hex::encode(expected_value_hash)
                    ),
                ));
            }
        } else if element.is_any_tree() && !element.is_non_empty_tree() {
            let element_hash = value_hash(value_bytes).value().to_owned();
            if matches!(
                element,
                Element::CountSumTree(None, ..)
                    | Element::ProvableCountTree(None, ..)
                    | Element::ProvableCountSumTree(None, ..)
            ) && hash == &element_hash
            {
                return Ok(());
            }
            let expected_value_hash = combine_hash(&element_hash, &NULL_HASH).value().to_owned();
            if hash != &expected_value_hash {
                return Err(invalid_proof(
                    query,
                    format!(
                        "V1 empty tree value hash mismatch at key {}: expected {}, got {}",
                        hex::encode(key),
                        hex::encode(hash),
                        hex::encode(expected_value_hash)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Verify an MMR lower layer proof and add results.
    /// The computed MMR root is the child hash of the parent Merk row
    /// (`combine_hash(value_hash || mmr_root)`) and is bound to `parent`
    /// before any leaf is reported.
    ///
    /// Root, then bind, then report: `mmr_lower_root` derives the root
    /// without touching `result` or `limit_left`, this function binds it,
    /// and only then does `report_mmr_leaves` add rows.
    #[allow(clippy::too_many_arguments)]
    fn verify_mmr_lower_layer<S: RowSink>(
        mmr_bytes: &[u8],
        element: &Element,
        parent: ParentCommitment<'_>,
        path: &[&[u8]],
        limit_left: &mut Option<u16>,
        result: &mut S,
        query: &PathQuery,
        report_contents: bool,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        let element_mmr_size = mmr_size_of(element, query)?;
        match Self::mmr_lower_root(mmr_bytes, element_mmr_size, query)? {
            // An empty MMR (mmr_size == 0) has no leaves to verify: its root
            // is the empty-MMR root hash ([0u8; 32]).
            MmrLowerRoot::Empty => parent.bind(query, &[0u8; 32]),
            MmrLowerRoot::Proved {
                root,
                leaves,
                mmr_size,
            } => {
                // Bind the root to the parent row
                // (combine_hash(value_hash || mmr_root)) before reporting
                // any leaf.
                parent.bind(query, &root)?;

                // Root only: the parent element is bound and this layer's
                // leaves are not reported, so there is no query at this path
                // to check completeness/succinctness against.
                if !report_contents {
                    return Ok(());
                }

                Self::report_mmr_leaves(
                    leaves,
                    mmr_size,
                    path,
                    limit_left,
                    result,
                    query,
                    grove_version,
                )
            }
        }
    }

    /// Decode an MMR lower layer proof, check it against the element's
    /// `mmr_size` (`mmr_size_of`), and compute its root. Reads neither the result set nor
    /// the row limit; the caller binds the root before anything is
    /// reported.
    fn mmr_lower_root(
        mmr_bytes: &[u8],
        element_mmr_size: u64,
        query: &PathQuery,
    ) -> Result<MmrLowerRoot, Error> {
        let mmr_proof = grovedb_merkle_mountain_range::MmrTreeProof::decode_from_slice(mmr_bytes)
            .map_err(|e| Error::CorruptedData(format!("{}", e)))?;

        // Cross-validate: proof's mmr_size must match the element's mmr_size
        if mmr_proof.mmr_size() != element_mmr_size {
            return Err(invalid_proof(
                query,
                format!(
                    "MMR proof mmr_size {} does not match element mmr_size {}",
                    mmr_proof.mmr_size(),
                    element_mmr_size
                ),
            ));
        }

        // An empty MMR (mmr_size == 0) has no leaves to verify: its root is
        // the empty-MMR root hash ([0u8; 32]), which the caller binds.
        if mmr_proof.leaves().is_empty() {
            if element_mmr_size == 0 {
                return Ok(MmrLowerRoot::Empty);
            }
            return Err(invalid_proof(
                query,
                format!(
                    "MMR proof contains no leaves but element mmr_size is {} (expected 0)",
                    element_mmr_size
                ),
            ));
        }

        // Compute root from the proof; the caller binds it to the parent row.
        let (mmr_root, verified_leaves) = mmr_proof
            .verify_and_get_root()
            .map_err(|e| invalid_proof(query, format!("{}", e)))?;
        Ok(MmrLowerRoot::Proved {
            root: mmr_root,
            leaves: verified_leaves,
            mmr_size: element_mmr_size,
        })
    }

    /// Report the proved MMR leaves the query selects at `path`, in the
    /// query's direction, charging `limit_left`. `verified_leaves` and
    /// `element_mmr_size` come from `mmr_lower_root`, whose root the caller
    /// has already bound to the parent row.
    fn report_mmr_leaves<S: RowSink>(
        verified_leaves: grovedb_merkle_mountain_range::VerifiedLeaves,
        element_mmr_size: u64,
        path: &[&[u8]],
        limit_left: &mut Option<u16>,
        result: &mut S,
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        // Get the sub-query items for this path to enforce succinctness.
        let sub_query = match query.query_items_at_path(path, grove_version)? {
            Some(sub_query) => sub_query,
            None => {
                return Err(invalid_proof(
                    query,
                    "MMR path not found in query".to_string(),
                ))
            }
        };

        // Completeness + soundness over position INTERVALS, never an
        // enumerated set. `element_mmr_size` is read from the parent
        // element bytes carried by the proof, which are only bound to the
        // trusted root when the caller checks the returned root, so nothing
        // below may cost work proportional to it: the interval build is
        // O(query items), and both checks are O(proved leaves). Each leaf
        // carries its value in the proof, and `verify_and_get_root` yields
        // one leaf per index, so the proof bytes bound that (issue #856).
        let leaf_count = grovedb_merkle_mountain_range::mmr_size_to_leaf_count(element_mmr_size);
        let expected_indices = PositionIntervals::from_query_items(&sub_query.items, leaf_count)?;
        let proved_indices: BTreeSet<u64> = verified_leaves.iter().map(|(idx, _)| *idx).collect();

        // Soundness: no unrequested leaves in the proof.
        let extra = expected_indices.extra_in(proved_indices.iter());
        if !extra.is_empty() {
            return Err(invalid_proof(
                query,
                format!(
                    "MMR proof contains unrequested leaf indices ({})",
                    extra.describe()
                ),
            ));
        }

        // Completeness: every requested leaf must be in the proof.
        let missing = expected_indices.missing_from(&proved_indices);
        if !missing.is_empty() {
            return Err(invalid_proof(
                query,
                format!(
                    "MMR proof missing requested leaf indices ({})",
                    missing.describe()
                ),
            ));
        }

        // Add each verified leaf to the result set, following the query
        // direction so a cap keeps the intended end of the range.
        let mut verified_leaves = verified_leaves;
        verified_leaves.sort_by_key(|(leaf_index, _)| *leaf_index);
        if !sub_query.left_to_right {
            verified_leaves.reverse();
        }
        for (leaf_index, value) in verified_leaves {
            let key = leaf_index.to_be_bytes().to_vec();
            let element = Element::new_item(value);
            let serialized = element.serialize(grove_version).map_err(|e| {
                Error::CorruptedData(format!("failed to serialize MMR leaf element: {}", e))
            })?;

            let path_key_optional_value = ProvedPathKeyOptionalValue {
                path: path.iter().map(|p| p.to_vec()).collect(),
                key,
                value: Some(serialized),
                proof: [0u8; 32],
            };
            result.push_row(path_key_optional_value, grove_version)?;

            limit_left
                .iter_mut()
                .for_each(|limit| *limit = limit.saturating_sub(1));
            if limit_left == &Some(0) {
                break;
            }
        }

        Ok(())
    }

    /// Verify a BulkAppendTree lower layer proof and add results.
    ///
    /// Verifies the proof's internal consistency, binds the computed child
    /// hash to `parent`, and only then reports the queried entries.
    #[allow(clippy::too_many_arguments)]
    fn verify_bulk_append_lower_layer<S: RowSink>(
        bulk_bytes: &[u8],
        element: &Element,
        parent: ParentCommitment<'_>,
        path: &[&[u8]],
        limit_left: &mut Option<u16>,
        result: &mut S,
        query: &PathQuery,
        report_contents: bool,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        let (total_count, height, is_bulk_append) = bulk_dims_of(element, query)?;
        let (child_hash, proof_result) = Self::decode_bulk_append_lower_layer(
            bulk_bytes,
            total_count,
            height,
            is_bulk_append,
            query,
        )?;
        parent.bind(query, &child_hash)?;
        if report_contents {
            Self::report_bulk_append_entries(
                &proof_result,
                path,
                limit_left,
                result,
                query,
                grove_version,
            )?;
        }
        Ok(())
    }

    /// Decode a `BulkAppendTree` / `CommitmentTree` bulk proof and check its
    /// internal consistency against the element's `total_count` and
    /// `height` (`bulk_dims_of`; `is_bulk_append` for a `BulkAppendTree`).
    /// Returns the bulk child hash and the verified proof result, from which
    /// nothing has been extracted yet.
    fn decode_bulk_append_lower_layer(
        bulk_bytes: &[u8],
        element_total_count: u64,
        element_height: u8,
        is_bulk_append: bool,
        query: &PathQuery,
    ) -> Result<
        (
            CryptoHash,
            grovedb_bulk_append_tree::BulkAppendTreeProofResult,
        ),
        Error,
    > {
        let bulk_proof =
            grovedb_bulk_append_tree::BulkAppendTreeProof::decode_from_slice(bulk_bytes)
                .map_err(|e| Error::CorruptedData(format!("{}", e)))?;

        let (bulk_state_root, proof_result) = bulk_proof
            .verify_and_compute_root(element_height, element_total_count)
            .map_err(|e| invalid_proof(query, format!("{}", e)))?;

        // An empty `BulkAppendTree` element chains through NULL_HASH, not the
        // domain-tagged empty state root: insert commits the element with a
        // NULL_HASH child hash (there is no bulk state until the first
        // append), and `verify_grovedb`'s integrity walk mirrors that. This
        // is sound because `verify_and_compute_root` above already rejected
        // any proof carrying data for a zero-count tree. `CommitmentTree` is
        // different — its insert commits `EMPTY_COMMITMENT_TREE_STATE_ROOT`,
        // which folds in the *computed* empty bulk root, so the CT wrapper
        // keeps the computed value.
        let child_hash = if element_total_count == 0 && is_bulk_append {
            NULL_HASH
        } else {
            bulk_state_root
        };
        Ok((child_hash, proof_result))
    }

    /// Report the entries of a verified bulk proof that the query selects
    /// at `path`, in the query's direction, charging `limit_left`.
    ///
    /// `proof_result.total_count` and `height` come from the parent element
    /// bytes carried by the proof. They are bound to the trusted root only
    /// when the caller checks the returned root, so nothing here may cost
    /// work proportional to them, or to what a chunk blob declares about
    /// itself:
    ///
    /// - completeness is checked first, against the positions the proof's
    ///   chunk indices and buffer entries cover, without decoding anything;
    /// - extraction then walks the query's intervals in its direction and
    ///   stops at the row limit, decoding only the entries it returns, from
    ///   chunks that must hold exactly `2^height` entries
    ///   (`BulkAppendTreeProofResult::values_in_ranges_directed`).
    ///
    /// So every extracted value is a distinct queried position, and a
    /// limited query decodes no more entries than it reports.
    fn report_bulk_append_entries<S: RowSink>(
        proof_result: &grovedb_bulk_append_tree::BulkAppendTreeProofResult,
        path: &[&[u8]],
        limit_left: &mut Option<u16>,
        result: &mut S,
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        let sub_query = match query.query_items_at_path(path, grove_version)? {
            Some(sub_query) => sub_query,
            None => {
                return Err(invalid_proof(
                    query,
                    "BulkAppendTree path not found in query".to_string(),
                ))
            }
        };

        // The positions the query selects, as intervals clipped to
        // `total_count` — never enumerated (issue #856).
        let expected_positions =
            PositionIntervals::from_query_items(&sub_query.items, proof_result.total_count)?;

        // Completeness: every position the query expects must be covered by
        // a proved chunk or buffer entry. Checked on spans, before anything
        // is decoded: O(intervals + proved chunks + buffer entries).
        let proved_spans = proof_result
            .proved_position_spans()
            .map_err(|e| invalid_proof(query, format!("{}", e)))?;
        let missing = expected_positions.missing_from_spans(&proved_spans);
        if !missing.is_empty() {
            return Err(invalid_proof(
                query,
                format!(
                    "BulkAppendTree proof missing requested positions ({})",
                    missing.describe()
                ),
            ));
        }

        // Only positions inside `expected_positions`, in the query direction,
        // so the cap keeps the LAST positions for a descending query.
        // Chunk-aligned proofs legitimately carry a superset of the queried
        // positions; the ones outside the query are never extracted. The row
        // loop below emits a row before it checks the limit, so the cap is
        // at least one.
        let values = proof_result
            .values_in_ranges_directed(
                expected_positions.ranges(),
                sub_query.left_to_right,
                limit_left.map(|limit| usize::from(limit.max(1))),
            )
            .map_err(|e| invalid_proof(query, format!("{}", e)))?;

        for (position, value) in values {
            let key = position.to_be_bytes().to_vec();
            let element = Element::new_item(value);
            let serialized = element.serialize(grove_version).map_err(|e| {
                Error::CorruptedData(format!(
                    "failed to serialize BulkAppendTree entry element: {}",
                    e
                ))
            })?;

            let path_key_optional_value = ProvedPathKeyOptionalValue {
                path: path.iter().map(|p| p.to_vec()).collect(),
                key,
                value: Some(serialized),
                proof: [0u8; 32],
            };
            result.push_row(path_key_optional_value, grove_version)?;

            limit_left
                .iter_mut()
                .for_each(|limit| *limit = limit.saturating_sub(1));
            if limit_left == &Some(0) {
                break;
            }
        }
        Ok(())
    }

    /// Verify a CommitmentTree lower layer proof and add results.
    ///
    /// The proof bytes are `sinsemilla_root (32 bytes) || bulk_append_proof`.
    /// Verifies the BulkAppendTree proof to get `bulk_state_root`, binds
    /// `blake3("ct_state" || sinsemilla_root || bulk_state_root)`, the child
    /// hash, to `parent`, and only then reports the queried entries.
    #[allow(clippy::too_many_arguments)]
    fn verify_commitment_tree_lower_layer<S: RowSink>(
        ct_bytes: &[u8],
        element: &Element,
        parent: ParentCommitment<'_>,
        path: &[&[u8]],
        limit_left: &mut Option<u16>,
        result: &mut S,
        query: &PathQuery,
        report_contents: bool,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        if ct_bytes.len() < 32 {
            return Err(invalid_proof(
                query,
                "CommitmentTree proof too short (missing sinsemilla_root)".to_string(),
            ));
        }

        let sinsemilla_root: [u8; 32] = ct_bytes[..32]
            .try_into()
            .expect("sinsemilla_root is exactly 32 bytes");
        let bulk_proof_bytes = &ct_bytes[32..];

        // Verify the bulk append proof to get bulk_state_root
        let (total_count, height, is_bulk_append) = bulk_dims_of(element, query)?;
        let (bulk_state_root, proof_result) = Self::decode_bulk_append_lower_layer(
            bulk_proof_bytes,
            total_count,
            height,
            is_bulk_append,
            query,
        )?;

        // Combine sinsemilla_root with bulk_state_root to produce the
        // child hash the parent row commits to.
        // Inlined from grovedb_commitment_tree::compute_commitment_tree_state_root
        // to avoid pulling the heavy orchard crate into the verify feature.
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ct_state");
        hasher.update(&sinsemilla_root);
        hasher.update(&bulk_state_root);
        let child_hash = *hasher.finalize().as_bytes();
        parent.bind(query, &child_hash)?;

        if report_contents {
            Self::report_bulk_append_entries(
                &proof_result,
                path,
                limit_left,
                result,
                query,
                grove_version,
            )?;
        }
        Ok(())
    }

    /// Verify a DenseAppendOnlyFixedSizeTree lower layer proof and add results.
    /// The computed dense root is the child hash of the parent Merk row and
    /// is bound to `parent` before any entry is reported.
    ///
    /// Root, then bind, then report: `dense_lower_root` derives the root
    /// (and, when reporting, the selected entries) without touching `result`
    /// or `limit_left`, this function binds it, and only then does
    /// `report_dense_entries` add rows.
    #[allow(clippy::too_many_arguments)]
    fn verify_dense_tree_lower_layer<S: RowSink>(
        dense_bytes: &[u8],
        element: &Element,
        parent: ParentCommitment<'_>,
        path: &[&[u8]],
        limit_left: &mut Option<u16>,
        result: &mut S,
        query: &PathQuery,
        report_contents: bool,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        let (element_count, element_height) = dense_dims_of(element, query)?;
        match Self::dense_lower_root(
            dense_bytes,
            element_count,
            element_height,
            path,
            query,
            report_contents,
            grove_version,
        )? {
            DenseLowerRoot::RootOnly(computed_root) => parent.bind(query, &computed_root),
            DenseLowerRoot::Selected {
                root,
                entries,
                left_to_right,
            } => {
                parent.bind(query, &root)?;
                Self::report_dense_entries(
                    entries,
                    left_to_right,
                    path,
                    limit_left,
                    result,
                    grove_version,
                )
            }
        }
    }

    /// Decode a dense lower layer proof (against the element's count and
    /// height, `dense_dims_of`) and compute its root. When
    /// `report_contents` is set, also check it against the query at `path`
    /// (completeness and soundness) and return the selected entries; the
    /// root computation differs between the two modes. Reads neither the
    /// result set nor the row limit; the caller binds the root before
    /// anything is reported.
    fn dense_lower_root(
        dense_bytes: &[u8],
        element_count: u16,
        element_height: u8,
        path: &[&[u8]],
        query: &PathQuery,
        report_contents: bool,
        grove_version: &GroveVersion,
    ) -> Result<DenseLowerRoot, Error> {
        let dense_proof =
            grovedb_dense_fixed_sized_merkle_tree::DenseTreeProof::decode_from_slice(dense_bytes)
                .map_err(|e| Error::CorruptedData(format!("{}", e)))?;

        // Root only: the caller binds the parent element to this root and
        // this layer's entries are not reported, so there is no query at
        // this path to check completeness/soundness against.
        if !report_contents {
            let (computed_root, _entries): ([u8; 32], Vec<(u16, Vec<u8>)>) = dense_proof
                .verify_and_get_root(element_height, element_count)
                .map_err(|e| invalid_proof(query, format!("{}", e)))?;
            return Ok(DenseLowerRoot::RootOnly(computed_root));
        }

        // Get the sub-query items for this path to build a query for
        // verify_for_query, which enforces both completeness and soundness.
        let sub_query = match query.query_items_at_path(path, grove_version)? {
            Some(sub_query) => sub_query,
            None => {
                return Err(invalid_proof(
                    query,
                    "DenseTree path not found in query".to_string(),
                ))
            }
        };
        let level_query = Query {
            items: sub_query.items.to_vec(),
            left_to_right: sub_query.left_to_right,
            ..Default::default()
        };

        // verify_for_query validates height+count from the authenticated Element,
        // checks completeness (all queried positions present) and soundness
        // (no unrequested positions), then returns only matching entries.
        // The entries it returns are bounded by the proof: each carries its
        // value, and duplicate or out-of-range positions are refused.
        let (computed_root, verified_entries): ([u8; 32], Vec<(u16, Vec<u8>)>) = dense_proof
            .verify_for_query(&level_query, element_height, element_count)
            .map_err(|e| invalid_proof(query, format!("{}", e)))?;
        Ok(DenseLowerRoot::Selected {
            root: computed_root,
            entries: verified_entries,
            left_to_right: sub_query.left_to_right,
        })
    }

    /// Report the dense entries `dense_lower_root` selected, in the query's
    /// direction (`left_to_right`), charging `limit_left`. The caller has
    /// already bound the root they were proved against to the parent row.
    fn report_dense_entries<S: RowSink>(
        verified_entries: Vec<(u16, Vec<u8>)>,
        left_to_right: bool,
        path: &[&[u8]],
        limit_left: &mut Option<u16>,
        result: &mut S,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        // Add each verified entry to the result set
        // Follow the query direction so a cap keeps the intended end of
        // the range.
        let mut verified_entries = verified_entries;
        verified_entries.sort_by_key(|(position, _)| *position);
        if !left_to_right {
            verified_entries.reverse();
        }
        for (position, value) in verified_entries {
            let key = position.to_be_bytes().to_vec();
            let elem = Element::new_item(value);
            let serialized = elem.serialize(grove_version).map_err(|e| {
                Error::CorruptedData(format!(
                    "failed to serialize DenseTree entry element: {}",
                    e
                ))
            })?;

            let path_key_optional_value = ProvedPathKeyOptionalValue {
                path: path.iter().map(|p| p.to_vec()).collect(),
                key,
                value: Some(serialized),
                proof: [0u8; 32],
            };
            result.push_row(path_key_optional_value, grove_version)?;

            limit_left
                .iter_mut()
                .for_each(|limit| *limit = limit.saturating_sub(1));
            if limit_left == &Some(0) {
                break;
            }
        }

        Ok(())
    }

    /// ╔══════════════════════════════════════════════════════════════════╗
    /// ║                  ⚠⚠⚠  DO NOT MODIFY V0 PROOFS  ⚠⚠⚠               ║
    /// ╠══════════════════════════════════════════════════════════════════╣
    /// ║ This is the V0 layer verifier. Grove versions v1 and v2 emit    ║
    /// ║ V0 proofs in production; the bytes they accept are part of      ║
    /// ║ those versions' wire format. Changing what V0 accepts (e.g.     ║
    /// ║ widening to count-offset paginated proofs, accepting new node   ║
    /// ║ kinds, relaxing rejection conditions) is consensus-breaking     ║
    /// ║ for already-deployed validators. Put new verifier features on   ║
    /// ║ V1 (`verify_layer_proof_v1`) behind a fresh grove version       ║
    /// ║ instead. See `prove_query_non_serialized_v0` in generate.rs     ║
    /// ║ for the full rationale.                                          ║
    /// ╚══════════════════════════════════════════════════════════════════╝
    pub(crate) fn verify_layer_proof<T>(
        layer_proof: &MerkOnlyLayerProof,
        prove_options: &ProveOptions,
        query: &PathQuery,
        limit_left: &mut Option<u16>,
        current_path: &[&[u8]],
        result: &mut Vec<T>,
        last_parent_tree_type: &mut Option<TreeFeatureType>,
        options: &VerifyOptions,
        current_depth: usize,
        grove_version: &GroveVersion,
    ) -> Result<CryptoHash, Error>
    where
        T: TryFromVersioned<ProvedPathKeyOptionalValue>,
        Error: From<<T as TryFromVersioned<ProvedPathKeyOptionalValue>>::Error>,
    {
        if current_depth > super::MAX_PROOF_DEPTH {
            return Err(Error::InvalidProof(
                query.clone(),
                "proof verification exceeded maximum depth limit".to_string(),
            ));
        }
        check_grovedb_v0!(
            "verify_layer_proof",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_layer_proof
        );

        let internal_query = query
            .query_items_at_path(current_path, grove_version)?
            .ok_or(Error::CorruptedPath(format!(
                "verify raw: path {} should be part of path_query {}",
                current_path
                    .iter()
                    .map(hex::encode)
                    .collect::<Vec<_>>()
                    .join("/"),
                query
            )))?;

        let level_query = Query {
            items: internal_query.items.to_vec(),
            left_to_right: internal_query.left_to_right,
            ..Default::default()
        };

        let (root_hash, merk_result) = level_query
            .execute_proof(
                &layer_proof.merk_proof,
                *limit_left,
                internal_query.left_to_right,
                0, // V0 proofs: allow items in value hash nodes for backwards compatibility
            )
            .unwrap()
            .map_err(|e| {
                Error::InvalidProof(
                    query.clone(),
                    format!("Invalid proof verification parameters: {}", e),
                )
            })?;

        #[cfg(feature = "proof_debug")]
        {
            println!(
                "\nDEBUG: Layer proof verification at path {:?}",
                current_path.iter().map(hex::encode).collect::<Vec<_>>()
            );
            println!("  Calculated root hash: {}", hex::encode(root_hash));
            if let Some(parent_type) = last_parent_tree_type {
                println!("  Parent tree type: {:?}", parent_type);
            }
        }
        #[cfg(feature = "proof_debug")]
        {
            println!(
                "current path {} \n merk result is {}",
                path_as_slices_hex_to_ascii(current_path),
                merk_result
            );
        }

        let mut verified_keys = BTreeSet::new();

        if merk_result.result_set.is_empty() {
            if prove_options.decrease_limit_on_empty_sub_query_result {
                limit_left
                    .iter_mut()
                    .for_each(|limit| *limit = limit.saturating_sub(1));
            }
        } else {
            for proved_key_value in merk_result.result_set {
                let mut path = current_path.to_vec();
                let key = &proved_key_value.key;
                let hash = &proved_key_value.proof;
                if let Some(value_bytes) = &proved_key_value.value {
                    // Look through NonCounted: dispatch on inner element. The
                    // serialized value bytes still hash to the on-disk value.
                    let element =
                        Element::deserialize(value_bytes, grove_version)?.into_underlying();

                    verified_keys.insert(key.clone());

                    if let Some(lower_layer) = layer_proof.lower_layers.get(key) {
                        #[cfg(feature = "proof_debug")]
                        {
                            println!("lower layer had key {}", hex_to_ascii(key));
                        }
                        match element {
                            // V0 is a frozen wire format and does not
                            // describe cidx descent (the prover rejects
                            // it for the same reason). The verifier
                            // refuses to fabricate a chain for an
                            // envelope shape V0 cannot produce.
                            Element::ProvableSumIndexedTree(..)
                            | Element::ProvableCountIndexedTree(..)
                            | Element::ProvableCountProvableSumIndexedTree(..) => {
                                return Err(Error::NotSupported(
                                    "V0 proofs do not support descent into any indexed-tree \
                                     element (ProvableSumIndexedTree / \
                                     ProvableCountIndexedTree / \
                                     ProvableCountProvableSumIndexedTree); use V1 or the \
                                     dedicated verify_indexed_count_top_k API"
                                        .to_string(),
                                ));
                            }
                            Element::Tree(Some(_), _)
                            | Element::SumTree(Some(_), ..)
                            | Element::BigSumTree(Some(_), ..)
                            | Element::CountTree(Some(_), ..)
                            | Element::CountSumTree(Some(_), ..)
                            | Element::ProvableCountTree(Some(_), ..)
                            | Element::ProvableCountSumTree(Some(_), ..)
                            | Element::ProvableSumTree(Some(_), ..)
                            | Element::ProvableCountProvableSumTree(Some(_), ..) => {
                                path.push(key);
                                *last_parent_tree_type = element.tree_feature_type();
                                if query.query_items_at_path(&path, grove_version)?.is_none() {
                                    // We are actually looking for the tree
                                    let path_key_optional_value =
                                        ProvedPathKeyOptionalValue::from_proved_key_value(
                                            path.iter().map(|p| p.to_vec()).collect(),
                                            proved_key_value,
                                        );
                                    #[cfg(feature = "proof_debug")]
                                    {
                                        println!(
                                            "pushing {} limit left after is {:?}",
                                            path_key_optional_value, limit_left
                                        );
                                    }
                                    result.push(
                                        path_key_optional_value
                                            .try_into_versioned(grove_version)?,
                                    );

                                    limit_left
                                        .iter_mut()
                                        .for_each(|limit| *limit = limit.saturating_sub(1));
                                    if limit_left == &Some(0) {
                                        break;
                                    }
                                } else {
                                    // Known limitation: this parent tree result
                                    // is pushed without decrementing limit_left.
                                    // Will be addressed by per-level limits
                                    // redesign.
                                    if query.should_add_parent_tree_at_path(
                                        current_path,
                                        grove_version,
                                    )? {
                                        let path_key_optional_value =
                                            ProvedPathKeyOptionalValue::from_proved_key_value(
                                                path.iter().map(|p| p.to_vec()).collect(),
                                                proved_key_value.clone(),
                                            );

                                        result.push(
                                            path_key_optional_value
                                                .try_into_versioned(grove_version)?,
                                        );
                                    }
                                    let lower_hash = Self::verify_layer_proof(
                                        lower_layer,
                                        prove_options,
                                        query,
                                        limit_left,
                                        &path,
                                        result,
                                        last_parent_tree_type,
                                        options,
                                        current_depth + 1,
                                        grove_version,
                                    )?;

                                    #[cfg(feature = "proof_debug")]
                                    {
                                        println!("\nDEBUG: Lower layer verification completed");
                                        println!(
                                            "  Path: {:?}",
                                            path.iter()
                                                .map(|p| hex_to_ascii(p))
                                                .collect::<Vec<_>>()
                                        );
                                        println!(
                                            "  Lower layer root hash: {}",
                                            hex::encode(lower_hash)
                                        );
                                        println!("  Parent tree type: {:?}", last_parent_tree_type);
                                    }
                                    let combined_root_hash =
                                        combine_hash(value_hash(value_bytes).value(), &lower_hash)
                                            .value()
                                            .to_owned();

                                    #[cfg(feature = "proof_debug")]
                                    {
                                        println!("\nDEBUG: Tree element verification");
                                        println!("  Key: {}", hex_to_ascii(key));
                                        println!(
                                            "  Element type: {:?}",
                                            element.tree_feature_type()
                                        );
                                        println!("  Value bytes: {}", hex::encode(value_bytes));
                                        println!(
                                            "  Value bytes hash: {}",
                                            hex::encode(value_hash(value_bytes).value())
                                        );
                                        println!("  Lower layer hash: {}", hex::encode(lower_hash));
                                        println!(
                                            "  Combined hash: {}",
                                            hex::encode(combined_root_hash)
                                        );
                                        println!("  Expected hash: {}", hex::encode(hash));
                                    }
                                    if hash != &combined_root_hash {
                                        return Err(Error::InvalidProof(
                                            query.clone(),
                                            format!(
                                                "Mismatch in lower layer hash, expected {}, got {}",
                                                hex::encode(hash),
                                                hex::encode(combined_root_hash)
                                            ),
                                        ));
                                    }
                                    if limit_left == &Some(0) {
                                        break;
                                    }
                                }
                            }
                            Element::Tree(None, _)
                            | Element::SumTree(None, ..)
                            | Element::BigSumTree(None, ..)
                            | Element::CountTree(None, ..)
                            | Element::CountSumTree(None, ..)
                            | Element::ProvableCountTree(None, ..)
                            | Element::ProvableCountSumTree(None, ..)
                            | Element::ProvableSumTree(None, ..)
                            | Element::ProvableCountProvableSumTree(None, ..)
                            | Element::CommitmentTree(..)
                            | Element::MmrTree(..)
                            | Element::BulkAppendTree(..)
                            | Element::DenseAppendOnlyFixedSizeTree(..)
                            | Element::PrivateDocumentStore(..)
                            | Element::SumItem(..)
                            | Element::Item(..)
                            | Element::ItemWithSumItem(..)
                            | Element::Reference(..)
                            | Element::ReferenceWithSumItem(..)
                            | Element::BidirectionalReference(..)
                            | Element::ItemWithBackwardsReferences(..)
                            | Element::SumItemWithBackwardsReferences(..)
                            | Element::ItemWithSumItemWithBackwardsReferences(..) => {
                                return Err(Error::InvalidProof(
                                    query.clone(),
                                    "Proof has lower layer for a non Tree.".to_string(),
                                ));
                            }
                            Element::NonCounted(_)
                            | Element::NotSummed(_)
                            | Element::NotCountedOrSummed(_) => {
                                unreachable!("unwrapped above")
                            }
                        }
                    } else if element.is_any_item()
                        || !internal_query.has_subquery_or_matching_in_path_on_key(key)
                            && (options.include_empty_trees_in_result
                                || !matches!(element, Element::Tree(None, _)))
                    {
                        // For empty trees in the result set (no lower layer
                        // proof), verify that the value_hash matches
                        // combine_hash(H(value), NULL_HASH). Without this
                        // check, an attacker could swap tree types (e.g.
                        // SumTree→Tree) in KVValueHash nodes without breaking
                        // the merk proof, since the value bytes are not part of
                        // the KVValueHash tree hash computation.
                        if element.is_any_tree() && !element.is_non_empty_tree() {
                            let expected_value_hash =
                                combine_hash(value_hash(value_bytes).value(), &NULL_HASH)
                                    .value()
                                    .to_owned();
                            if hash != &expected_value_hash {
                                return Err(Error::InvalidProof(
                                    query.clone(),
                                    format!(
                                        "empty tree value hash mismatch at key {}: expected \
                                         {}, got {}",
                                        hex::encode(key),
                                        hex::encode(hash),
                                        hex::encode(expected_value_hash)
                                    ),
                                ));
                            }
                        }

                        // V0 proofs do not require
                        // KVValueHashFeatureTypeWithChildHash for non-empty
                        // trees without subqueries — that check is V1-only.

                        let path_key_optional_value =
                            ProvedPathKeyOptionalValue::from_proved_key_value(
                                path.iter().map(|p| p.to_vec()).collect(),
                                proved_key_value,
                            );
                        #[cfg(feature = "proof_debug")]
                        {
                            println!(
                                "pushing {} limit left after is {:?}",
                                path_key_optional_value, limit_left
                            );
                        }
                        result.push(path_key_optional_value.try_into_versioned(grove_version)?);

                        limit_left
                            .iter_mut()
                            .for_each(|limit| *limit = limit.saturating_sub(1));
                        if limit_left == &Some(0) {
                            break;
                        }
                    } else if element.is_non_empty_tree()
                        && internal_query.has_subquery_or_matching_in_path_on_key(key)
                    {
                        return Err(Error::InvalidProof(
                            query.clone(),
                            format!(
                                "Proof is missing lower layer for non-empty tree at key {}",
                                hex::encode(key),
                            ),
                        ));
                    }
                }
            }
        }

        // Succinctness check: reject proof if it contains lower layers
        // for keys that the query did not require
        if options.verify_proof_succinctness {
            for key in layer_proof.lower_layers.keys() {
                if !verified_keys.contains(key) {
                    return Err(Error::InvalidProof(
                        query.clone(),
                        format!(
                            "Proof contains extra lower layer for key {} not required by query",
                            hex::encode(key),
                        ),
                    ));
                }
            }
        }

        Ok(root_hash)
    }

    /// Verifies a proof against a path query with succinctness checks,
    /// returning the root hash and deserialized results.
    pub fn verify_query(
        proof: &[u8],
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Vec<PathKeyOptionalElementTrio>), Error> {
        check_grovedb_v0!(
            "verify_query",
            grove_version.grovedb_versions.operations.proof.verify_query
        );
        Self::verify_query_with_options(
            proof,
            query,
            VerifyOptions {
                absence_proofs_for_non_existing_searched_keys: false,
                verify_proof_succinctness: true,
                include_empty_trees_in_result: false,
            },
            grove_version,
        )
    }

    /// Verifies a proof against a path query without succinctness checks,
    /// allowing the proof to contain extra data beyond what the query requires.
    pub fn verify_subset_query(
        proof: &[u8],
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Vec<PathKeyOptionalElementTrio>), Error> {
        check_grovedb_v0!(
            "verify_subset_query",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_subset_query
        );
        Self::verify_query_with_options(
            proof,
            query,
            VerifyOptions {
                absence_proofs_for_non_existing_searched_keys: false,
                verify_proof_succinctness: false,
                include_empty_trees_in_result: false,
            },
            grove_version,
        )
    }

    /// Verify a proof produced by [`GroveDb::prove_bulk_position_range`],
    /// returning `(root_hash, page)` for the position range
    /// `[start, start + limit)` of the append-only, BulkAppendTree-backed
    /// element at `path`/`key` (`Element::BulkAppendTree` or
    /// `Element::CommitmentTree`).
    ///
    /// The page's entries are the `(position, value)` pairs, ascending and
    /// contiguous from `start`, clamped to the element's authenticated
    /// `total_count` (returned in
    /// [`RangePage::total_count`](grovedb_bulk_append_tree::RangePage::total_count)).
    /// Completeness is enforced: a proof missing any requested position below
    /// `total_count` is rejected. Absence beyond the end falls out of the
    /// provable count — positions `>= total_count` do not exist — so a page
    /// shorter than `limit` means the scan caught up with the tip; no
    /// per-position absence proofs are involved.
    ///
    /// Both the entries and `total_count` are extracted from the same proof
    /// bytes: the entries by verifying the canonical
    /// [`PathQuery::new_bulk_position_range`] query, and `total_count` by
    /// subset-verifying the element itself, whose serialized bytes are bound
    /// to the parent Merk (and through it to `root_hash`).
    pub fn verify_bulk_position_range_proof(
        proof: &[u8],
        path: Vec<Vec<u8>>,
        key: &[u8],
        start: u64,
        limit: u16,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, grovedb_bulk_append_tree::RangePage), Error> {
        check_grovedb_v0!(
            "verify_bulk_position_range_proof",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_bulk_position_range_proof
        );
        // 1. Verify the range entries against the canonical range query.
        //    Succinctness cannot be required: range proofs are chunk-aligned
        //    and intentionally carry whole chunk blobs — a superset of the
        //    queried positions.
        let range_query =
            PathQuery::new_bulk_position_range(path.clone(), key.to_vec(), start, limit);
        let (root_hash, results) = Self::verify_query_with_options(
            proof,
            &range_query,
            VerifyOptions {
                absence_proofs_for_non_existing_searched_keys: false,
                verify_proof_succinctness: false,
                include_empty_trees_in_result: false,
            },
            grove_version,
        )?;

        // 2. Extract the element's authenticated total_count from the same
        //    proof bytes via a single-key subset query on the element itself.
        let element_query = PathQuery::new_single_key(path, key.to_vec());
        let (element_root_hash, element_results) =
            Self::verify_subset_query(proof, &element_query, grove_version)?;
        if element_root_hash != root_hash {
            return Err(Error::InvalidProof(
                range_query,
                "range and element sub-proofs derived different root hashes".to_string(),
            ));
        }
        let total_count = match bound_append_only_total_count(&element_results, key) {
            Ok(total_count) => total_count,
            Err(message) => {
                return Err(Error::InvalidProof(range_query, message.to_string()));
            }
        };

        // 3. Map the results to (position, value) entries and require the
        //    page to be exactly [start, min(start + limit, total_count)) —
        //    ascending, contiguous, complete. The lower-layer verification
        //    already enforces completeness; this re-check keeps the helper's
        //    guarantee independent of that layer's internals.
        let end = start
            .saturating_add(limit as u64)
            .min(total_count)
            .max(start);
        let entries = bulk_range_entries(&results, start, end, &range_query)?;

        Ok((
            root_hash,
            grovedb_bulk_append_tree::RangePage {
                entries,
                total_count,
            },
        ))
    }

    /// The point of this query is to get the parent tree information which will
    /// be present because we are querying in a subtree
    pub fn verify_query_get_parent_tree_info(
        proof: &[u8],
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, TreeFeatureType, Vec<PathKeyOptionalElementTrio>), Error> {
        check_grovedb_v0!(
            "verify_query_get_parent_tree_info",
            grove_version.grovedb_versions.operations.proof.verify_query
        );
        Self::verify_query_get_parent_tree_info_with_options(
            proof,
            query,
            VerifyOptions {
                absence_proofs_for_non_existing_searched_keys: false,
                verify_proof_succinctness: true,
                include_empty_trees_in_result: false,
            },
            grove_version,
        )
    }

    /// The point of this query is to get the parent tree information which will
    /// be present because we are querying in a subtree
    pub fn verify_subset_query_get_parent_tree_info(
        proof: &[u8],
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, TreeFeatureType, Vec<PathKeyOptionalElementTrio>), Error> {
        check_grovedb_v0!(
            "verify_subset_query_get_parent_tree_info",
            grove_version.grovedb_versions.operations.proof.verify_query
        );
        Self::verify_query_get_parent_tree_info_with_options(
            proof,
            query,
            VerifyOptions {
                absence_proofs_for_non_existing_searched_keys: false,
                verify_proof_succinctness: false,
                include_empty_trees_in_result: false,
            },
            grove_version,
        )
    }

    /// Verifies a proof that includes absence proofs for non-existing searched
    /// keys, returning the root hash and deserialized results.
    pub fn verify_query_with_absence_proof(
        proof: &[u8],
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Vec<PathKeyOptionalElementTrio>), Error> {
        check_grovedb_v0!(
            "verify_query_with_absence_proof",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_query_with_absence_proof
        );
        Self::verify_query_with_options(
            proof,
            query,
            VerifyOptions {
                absence_proofs_for_non_existing_searched_keys: true,
                verify_proof_succinctness: true,
                include_empty_trees_in_result: false,
            },
            grove_version,
        )
    }

    /// Verifies a subset proof that includes absence proofs for non-existing
    /// searched keys, without succinctness checks.
    pub fn verify_subset_query_with_absence_proof(
        proof: &[u8],
        query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Vec<PathKeyOptionalElementTrio>), Error> {
        check_grovedb_v0!(
            "verify_subset_query_with_absence_proof",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_subset_query_with_absence_proof
        );
        Self::verify_query_with_options(
            proof,
            query,
            VerifyOptions {
                absence_proofs_for_non_existing_searched_keys: true,
                verify_proof_succinctness: false,
                include_empty_trees_in_result: false,
            },
            grove_version,
        )
    }

    /// Verify subset proof with a chain of path query functions.
    /// After subset verification with the first path query, the result if
    /// passed to the next path query generation function which generates a
    /// new path query Apply the new path query, and pass the result to the
    /// next ... This is useful for verifying proofs with multiple path
    /// queries that depend on one another.
    pub fn verify_query_with_chained_path_queries<C>(
        proof: &[u8],
        first_query: &PathQuery,
        chained_path_queries: Vec<C>,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, Vec<Vec<PathKeyOptionalElementTrio>>), Error>
    where
        C: Fn(Vec<PathKeyOptionalElementTrio>) -> Option<PathQuery>,
    {
        check_grovedb_v0!(
            "verify_query_with_chained_path_queries",
            grove_version
                .grovedb_versions
                .operations
                .proof
                .verify_query_with_chained_path_queries
        );
        let mut results = vec![];

        let (last_root_hash, elements) =
            Self::verify_subset_query(proof, first_query, grove_version)?;
        results.push(elements);

        // Process the chained path queries
        Self::process_chained_path_queries(
            proof,
            last_root_hash,
            chained_path_queries,
            grove_version,
            &mut results,
        )?;

        Ok((last_root_hash, results))
    }

    /// Processes each chained path query and verifies it.
    pub(in crate::operations::proof) fn process_chained_path_queries<C>(
        proof: &[u8],
        last_root_hash: CryptoHash,
        chained_path_queries: Vec<C>,
        grove_version: &GroveVersion,
        results: &mut Vec<Vec<PathKeyOptionalElementTrio>>,
    ) -> Result<(), Error>
    where
        C: Fn(Vec<PathKeyOptionalElementTrio>) -> Option<PathQuery>,
    {
        for path_query_generator in chained_path_queries {
            let new_path_query = path_query_generator(results[results.len() - 1].clone()).ok_or(
                Error::InvalidInput("one of the path query generators returns no path query"),
            )?;

            let (new_root_hash, new_elements) =
                Self::verify_subset_query(proof, &new_path_query, grove_version)?;

            if new_root_hash != last_root_hash {
                return Err(Error::InvalidProof(
                    new_path_query,
                    format!(
                        "Root hash for different path queries do not match, first is {}, this one \
                         is {}",
                        hex::encode(last_root_hash),
                        hex::encode(new_root_hash)
                    ),
                ));
            }

            results.push(new_elements);
        }

        Ok(())
    }

    /// Verifies a trunk chunk proof and returns a `GroveTrunkQueryResult`.
    ///
    /// This method verifies a proof generated by `prove_trunk_chunk`, walking
    /// through the path layers and verifying each one. At the target tree,
    /// it decodes and executes the trunk proof to extract the elements and
    /// leaf keys.
    ///
    /// # Arguments
    /// * `proof` - The serialized proof bytes
    /// * `query` - The path trunk chunk query (used to navigate the proof)
    /// * `grove_version` - The GroveDB version for element deserialization
    ///
    /// # Returns
    /// A tuple of:
    /// * `CryptoHash` - The root hash of the entire GroveDB
    /// * `GroveTrunkQueryResult` - The verified result with elements, leaf
    ///   keys, chunk depths, and tree depth
    pub fn verify_trunk_chunk_proof(
        proof: &[u8],
        query: &PathTrunkChunkQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, GroveTrunkQueryResult), Error> {
        let grovedb_proof = super::decode_grovedb_proof_canonical(proof)?;

        match grovedb_proof {
            GroveDBProof::V0(proof_v0) => {
                Self::verify_trunk_chunk_proof_v0(&proof_v0, query, grove_version)
            }
            GroveDBProof::V1(proof_v1) => {
                Self::verify_trunk_chunk_proof_v1(&proof_v1, query, grove_version)
            }
        }
    }

    fn verify_trunk_chunk_proof_v0(
        proof: &GroveDBProofV0,
        query: &PathTrunkChunkQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, GroveTrunkQueryResult), Error> {
        // Collect layer info as we walk down the path for later verification
        let mut layer_infos: Vec<TrunkLayerInfo> = Vec::new();

        let mut current_layer = &proof.root_layer;
        let mut current_path: Vec<Vec<u8>> = Vec::new();
        let mut count: Option<u64> = None;
        let mut grovedb_root_hash: Option<CryptoHash> = None;

        // Walk through each path segment, verifying layer proofs. An index
        // loop over a per-segment helper; the first failing segment's
        // error ends the walk.
        let mut failure = None;
        let mut failed = false;
        let mut i = 0;
        while i < query.path.len() && !failed {
            let is_last = i == query.path.len() - 1;
            match Self::trunk_path_layer_v0(
                current_layer,
                &query.path[i],
                i,
                is_last,
                &current_path,
                grove_version,
            ) {
                Ok((layer_info, layer_count, lower_layer)) => {
                    // Store the root hash of the first layer as the GroveDB root hash
                    if i == 0 {
                        grovedb_root_hash = Some(layer_info.layer_root_hash);
                    }
                    // On the last path segment, the count of the CountTree element
                    if is_last {
                        count = layer_count;
                    }
                    layer_infos.push(layer_info);
                    current_layer = lower_layer;
                    current_path.push(query.path[i].clone());
                }
                Err(e) => {
                    failure = Some(e);
                    failed = true;
                }
            }
            i += 1;
        }
        if let Some(e) = failure {
            return Err(e);
        }

        // Ensure we got a count from the element
        let count = match count {
            Some(count) => count,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    "Could not extract count from path - target is not a count tree element"
                        .to_string(),
                ));
            }
        };

        // Empty tree — return an empty result.
        // The parent layers' merk proofs already verified the hash chain integrity,
        // including the element's value_hash. The count is authenticated by the
        // parent proof. With count==0 there is no child data to verify against.
        if count == 0 {
            let grovedb_root_hash = match grovedb_root_hash {
                Some(hash) => hash,
                None => {
                    return Err(Error::InvalidProof(
                        PathQuery::new_unsized(Vec::new(), Query::default()),
                        "Empty path - no root hash computed".to_string(),
                    ));
                }
            };

            return Ok((
                grovedb_root_hash,
                GroveTrunkQueryResult {
                    elements: BTreeMap::new(),
                    leaf_keys: BTreeMap::new(),
                    chunk_depths: vec![],
                    max_tree_depth: 0,
                    tree: grovedb_merk::proofs::tree::Tree::from(Node::Hash(NULL_HASH)),
                },
            ));
        }

        let tree_depth = calculate_max_tree_depth_from_count(count);
        let chunk_depths = match calculate_chunk_depths(tree_depth, query.max_depth) {
            Ok(chunk_depths) => chunk_depths,
            Err(e) => {
                return Err(Error::CorruptedData(format!(
                    "invalid chunk depth parameters: {}",
                    e
                )));
            }
        };

        // Now we're at the target layer - decode and execute the trunk proof
        let mut decoder = Decoder::new(&current_layer.merk_proof);
        let ops = match decode_chunk_ops(&mut decoder) {
            Ok(ops) => ops,
            Err(e) => {
                return Err(Error::CorruptedData(format!(
                    "Failed to decode trunk proof: {}",
                    e
                )));
            }
        };
        if decoder.remaining_bytes() > 0 {
            return Err(Error::CorruptedData(format!(
                "Trunk proof has {} unconsumed trailing bytes",
                decoder.remaining_bytes()
            )));
        }

        // Execute the proof to build the tree structure and get its root hash
        // Use collapse=false to preserve the full tree structure for element extraction
        let target_tree = match execute_chunk_ops(&ops).unwrap() {
            Ok(target_tree) => target_tree,
            Err(e) => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    format!("Failed to execute trunk proof: {}", e),
                ));
            }
        };

        let lower_hash = target_tree.hash().unwrap();

        // Verify the cryptographic chain from trunk up through all layers
        // Walk backwards through the layer infos, verifying each step
        check_trunk_chain_v0(&layer_infos, lower_hash, &current_path)?;

        // Extract elements and leaf keys from the proof tree
        // The max depth for the trunk is the first chunk depth
        let max_depth = first_chunk_depth(&chunk_depths);
        let mut elements = BTreeMap::new();
        let mut leaf_keys = BTreeMap::new();
        Self::extract_elements_and_leaf_keys(
            &target_tree,
            &mut elements,
            &mut leaf_keys,
            0,
            max_depth,
            grove_version,
        )?;

        let grovedb_root_hash = match grovedb_root_hash {
            Some(hash) => hash,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(Vec::new(), Query::default()),
                    "Empty path - no root hash computed".to_string(),
                ));
            }
        };

        let trunk_result = GroveTrunkQueryResult {
            elements,
            leaf_keys,
            chunk_depths,
            max_tree_depth: tree_depth,
            tree: target_tree,
        };

        Ok((grovedb_root_hash, trunk_result))
    }

    /// One path segment of the V0 trunk walk (the body of its path loop):
    /// prove `path_segment` in `layer`, then return the layer's record,
    /// the count of the target element (meaningful only when `is_last`),
    /// and the next layer. Errors in the order the loop raised them.
    fn trunk_path_layer_v0<'a>(
        layer: &'a MerkOnlyLayerProof,
        path_segment: &Vec<u8>,
        i: usize,
        is_last: bool,
        current_path: &[Vec<u8>],
        grove_version: &GroveVersion,
    ) -> Result<(TrunkLayerInfo, Option<u64>, &'a MerkOnlyLayerProof), Error> {
        // Create a simple key query for this path segment
        let key_query = single_key_query(path_segment);

        // Execute the proof to verify and get the root hash
        let (layer_root_hash, result) = match key_query
            .execute_proof(
                &layer.merk_proof,
                None,
                true,
                0, // V0 proof: allow items in value hash nodes
            )
            .unwrap()
        {
            Ok(executed) => executed,
            Err(e) => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query.clone()),
                    format!("Invalid proof at path layer {}: {}", i, e),
                ));
            }
        };

        // Find the element for this key in the result set
        let (found_value_bytes, found_hash, count) =
            Self::trunk_segment_row(&result.result_set, path_segment, is_last, grove_version)?;

        let value_bytes = match found_value_bytes {
            Some(value_bytes) => value_bytes,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query.clone()),
                    format!(
                        "Path segment {} not found in proof result",
                        hex::encode(path_segment)
                    ),
                ));
            }
        };

        let expected_hash = match found_hash {
            Some(expected_hash) => expected_hash,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query.clone()),
                    format!(
                        "No hash found for path segment {}",
                        hex::encode(path_segment)
                    ),
                ));
            }
        };

        // Move to the next layer
        let lower_layer = match layer.lower_layers.get(path_segment) {
            Some(lower_layer) => lower_layer,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query),
                    format!(
                        "Missing lower layer for path segment {}",
                        hex::encode(path_segment)
                    ),
                ));
            }
        };

        Ok((
            TrunkLayerInfo {
                value_bytes,
                expected_hash,
                layer_root_hash,
            },
            count,
            lower_layer,
        ))
    }

    /// The row a trunk path layer proved for `path_segment`: the first row
    /// with that key, as `(value bytes, value hash, count)`. `None`s when
    /// there is no such row (or it carries no value). When `is_last`, the
    /// value is deserialized and the count read from it, so a value that
    /// fails to deserialize is an error, as in the loop this came from.
    fn trunk_segment_row(
        rows: &[ProvedKeyOptionalValue],
        path_segment: &[u8],
        is_last: bool,
        grove_version: &GroveVersion,
    ) -> Result<(Option<Vec<u8>>, Option<CryptoHash>, Option<u64>), Error> {
        let found = super::aggregate_common::first_row_with_key(rows, path_segment);
        if found == rows.len() {
            return Ok((None, None, None));
        }
        let row = &rows[found];
        match &row.value {
            Some(value_bytes) => {
                let count = if is_last {
                    let element = Element::deserialize(value_bytes, grove_version)?;
                    Self::extract_count_from_element(&element)
                } else {
                    None
                };
                Ok((Some(value_bytes.clone()), Some(row.proof), count))
            }
            None => Ok((None, Some(row.proof), None)),
        }
    }

    /// V1 trunk chunk proof verifier.
    ///
    /// Works like V0 but uses `LayerProof`/`ProofBytes::Merk` types and,
    /// critically, **always** runs the `combine_hash(value_hash, lower_hash)`
    /// verification chain -- even when `count == 0`. This prevents an
    /// attacker from forging the element bytes (e.g. setting count to 0)
    /// while keeping the original `value_hash` intact in a `KVValueHash` node,
    /// which would let the V0 fast-path skip the hash verification.
    fn verify_trunk_chunk_proof_v1(
        proof: &GroveDBProofV1,
        query: &PathTrunkChunkQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, GroveTrunkQueryResult), Error> {
        if query.path.is_empty() {
            return Err(Error::InvalidProof(
                PathQuery::new_unsized(Vec::new(), Query::default()),
                "V1 trunk: empty path - no root hash computed".to_string(),
            ));
        }

        // Collect layer info as we walk down the path for later verification
        let mut layer_infos: Vec<TrunkLayerInfo> = Vec::new();

        let mut current_layer = &proof.root_layer;
        let mut current_path: Vec<Vec<u8>> = Vec::new();
        let mut count: Option<u64> = None;
        let mut grovedb_root_hash: CryptoHash = [0u8; 32];

        // Walk through each path segment, verifying layer proofs. An index
        // loop over a per-segment helper; the first failing segment's
        // error ends the walk.
        let mut failure = None;
        let mut failed = false;
        let mut i = 0;
        while i < query.path.len() && !failed {
            let is_last = i == query.path.len() - 1;
            match Self::trunk_path_layer_v1(
                current_layer,
                &query.path[i],
                i,
                is_last,
                &current_path,
                grove_version,
            ) {
                Ok((layer_info, layer_count, lower_layer)) => {
                    if i == 0 {
                        grovedb_root_hash = layer_info.layer_root_hash;
                    }
                    if is_last {
                        count = layer_count;
                    }
                    layer_infos.push(layer_info);
                    current_layer = lower_layer;
                    current_path.push(query.path[i].clone());
                }
                Err(e) => {
                    failure = Some(e);
                    failed = true;
                }
            }
            i += 1;
        }
        if let Some(e) = failure {
            return Err(e);
        }

        let count = match count {
            Some(count) => count,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    "V1 trunk: could not extract count from path - target is not a count tree \
                     element"
                        .to_string(),
                ));
            }
        };

        // Extract merk bytes from the target layer
        let target_merk_bytes = match &current_layer.merk_proof {
            ProofBytes::Merk(bytes) => bytes,
            _ => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    "V1 trunk proof: expected Merk proof bytes at target layer".to_string(),
                ));
            }
        };

        // Determine the lower hash -- always computed, even for empty trees.
        // This is the critical security fix: V0 skipped this for count==0.
        let (lower_hash, target_tree_opt) =
            Self::trunk_target_v1(target_merk_bytes, count, query, &current_path)?;

        // ALWAYS verify the cryptographic chain from trunk up through all layers.
        // This is the security fix: V0 skipped this when count==0, allowing
        // an attacker to forge the element bytes while keeping the original
        // value_hash in a KVValueHash node.
        check_trunk_chain_v1(&layer_infos, lower_hash, &current_path)?;

        // Handle empty tree early return (AFTER hash verification)
        if count == 0 {
            return Ok((
                grovedb_root_hash,
                GroveTrunkQueryResult {
                    elements: BTreeMap::new(),
                    leaf_keys: BTreeMap::new(),
                    chunk_depths: vec![],
                    max_tree_depth: 0,
                    tree: grovedb_merk::proofs::tree::Tree::from(Node::Hash(NULL_HASH)),
                },
            ));
        }

        // Non-empty tree: extract elements
        let target_tree = target_tree_opt.unwrap();
        let tree_depth = calculate_max_tree_depth_from_count(count);
        let chunk_depths = match calculate_chunk_depths(tree_depth, query.max_depth) {
            Ok(chunk_depths) => chunk_depths,
            Err(e) => {
                return Err(Error::CorruptedData(format!(
                    "invalid chunk depth parameters: {}",
                    e
                )));
            }
        };

        let max_depth = first_chunk_depth(&chunk_depths);
        let mut elements = BTreeMap::new();
        let mut leaf_keys = BTreeMap::new();
        Self::extract_elements_and_leaf_keys(
            &target_tree,
            &mut elements,
            &mut leaf_keys,
            0,
            max_depth,
            grove_version,
        )?;

        let trunk_result = GroveTrunkQueryResult {
            elements,
            leaf_keys,
            chunk_depths,
            max_tree_depth: tree_depth,
            tree: target_tree,
        };

        Ok((grovedb_root_hash, trunk_result))
    }

    /// One path segment of the V1 trunk walk (the body of its path loop):
    /// require Merk bytes, prove `path_segment` in `layer`, then return the
    /// layer's record, the count of the target element (meaningful only
    /// when `is_last`), and the next layer. Errors in the order the loop
    /// raised them.
    fn trunk_path_layer_v1<'a>(
        layer: &'a LayerProof,
        path_segment: &Vec<u8>,
        i: usize,
        is_last: bool,
        current_path: &[Vec<u8>],
        grove_version: &GroveVersion,
    ) -> Result<(TrunkLayerInfo, Option<u64>, &'a LayerProof), Error> {
        let key_query = single_key_query(path_segment);

        // Extract merk bytes from ProofBytes::Merk
        let merk_proof_bytes = match &layer.merk_proof {
            ProofBytes::Merk(bytes) => bytes,
            _ => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query),
                    "V1 trunk proof: expected Merk proof bytes at path layer".to_string(),
                ));
            }
        };

        let (layer_root_hash, result) = match key_query
            .execute_proof(merk_proof_bytes, None, true, PROOF_VERSION_LATEST)
            .unwrap()
        {
            Ok(executed) => executed,
            Err(e) => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query.clone()),
                    format!("Invalid V1 trunk proof at path layer {}: {}", i, e),
                ));
            }
        };

        let (found_value_bytes, found_hash, count) =
            Self::trunk_segment_row(&result.result_set, path_segment, is_last, grove_version)?;

        let value_bytes = match found_value_bytes {
            Some(value_bytes) => value_bytes,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query.clone()),
                    format!(
                        "V1 trunk: path segment {} not found in proof result",
                        hex::encode(path_segment)
                    ),
                ));
            }
        };

        let expected_hash = match found_hash {
            Some(expected_hash) => expected_hash,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query.clone()),
                    format!(
                        "V1 trunk: no hash found for path segment {}",
                        hex::encode(path_segment)
                    ),
                ));
            }
        };

        let lower_layer = match layer.lower_layers.get(path_segment) {
            Some(lower_layer) => lower_layer,
            None => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), key_query),
                    format!(
                        "V1 trunk: missing lower layer for path segment {}",
                        hex::encode(path_segment)
                    ),
                ));
            }
        };

        Ok((
            TrunkLayerInfo {
                value_bytes,
                expected_hash,
                layer_root_hash,
            },
            count,
            lower_layer,
        ))
    }

    /// The V1 trunk's target layer: for `count == 0` its bytes must be
    /// empty and its hash is `NULL_HASH`; otherwise the ops are decoded
    /// (all of them, no trailing bytes) and executed into the target tree,
    /// whose hash is the lower hash of the chain.
    fn trunk_target_v1(
        target_merk_bytes: &[u8],
        count: u64,
        query: &PathTrunkChunkQuery,
        current_path: &[Vec<u8>],
    ) -> Result<(CryptoHash, Option<grovedb_merk::proofs::tree::Tree>), Error> {
        if count == 0 {
            if !target_merk_bytes.is_empty() {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), Query::default()),
                    format!(
                        "V1 trunk: target layer has {} bytes but count is 0 (expected empty)",
                        target_merk_bytes.len()
                    ),
                ));
            }
            return Ok((NULL_HASH, None));
        }
        let tree_depth = calculate_max_tree_depth_from_count(count);
        if let Err(e) = calculate_chunk_depths(tree_depth, query.max_depth) {
            return Err(Error::CorruptedData(format!(
                "invalid chunk depth parameters: {}",
                e
            )));
        }

        let mut decoder = Decoder::new(target_merk_bytes);
        let ops = match decode_chunk_ops(&mut decoder) {
            Ok(ops) => ops,
            Err(e) => {
                return Err(Error::CorruptedData(format!(
                    "Failed to decode V1 trunk proof: {}",
                    e
                )));
            }
        };
        if decoder.remaining_bytes() > 0 {
            return Err(Error::CorruptedData(format!(
                "V1 trunk proof has {} unconsumed trailing bytes",
                decoder.remaining_bytes()
            )));
        }

        let target_tree = match execute_chunk_ops(&ops).unwrap() {
            Ok(target_tree) => target_tree,
            Err(e) => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.to_vec(), Query::default()),
                    format!("Failed to execute V1 trunk proof: {}", e),
                ));
            }
        };

        let hash = target_tree.hash().unwrap();
        Ok((hash, Some(target_tree)))
    }

    /// Recursively extract elements and leaf keys from a proof tree.
    ///
    /// Elements are nodes with key-value data that can be deserialized.
    /// Leaf keys are nodes that have at least one `Node::Hash` child, mapped
    /// to their LeafInfo (hash + optional count for branch verification).
    ///
    /// # Arguments
    /// * `tree` - The proof tree to extract from
    /// * `elements` - Output map of key -> Element
    /// * `leaf_keys` - Output map of key -> LeafInfo (hash + count)
    /// * `current_depth` - Current depth in the tree (0 = root)
    /// * `max_depth` - Maximum allowed depth (nodes beyond this should be Hash)
    /// * `grove_version` - Version for Element deserialization
    fn extract_elements_and_leaf_keys(
        tree: &grovedb_merk::proofs::tree::Tree,
        elements: &mut BTreeMap<Vec<u8>, Element>,
        leaf_keys: &mut BTreeMap<Vec<u8>, crate::query::LeafInfo>,
        current_depth: usize,
        max_depth: usize,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        // Check that we haven't exceeded the max depth
        if current_depth > max_depth {
            return Err(Error::InvalidProof(
                PathQuery::new_unsized(Vec::new(), Query::default()),
                format!(
                    "Trunk proof exceeds max depth: current depth {} > max depth {}",
                    current_depth, max_depth
                ),
            ));
        }

        // Check for inconsistent depth: if one child is Hash and the other exists
        // but is not Hash, the proof has inconsistent truncation depth
        let left_is_hash = child_is_hash(&tree.left);
        let right_is_hash = child_is_hash(&tree.right);

        // If both children exist, they must both be Hash or both be non-Hash
        if let (Some(left_hash), Some(right_hash)) = (left_is_hash, right_is_hash)
            && left_hash != right_hash
        {
            return Err(Error::InvalidProof(
                PathQuery::new_unsized(Vec::new(), Query::default()),
                "Inconsistent trunk proof depth: one child is Hash while the other is not"
                    .to_string(),
            ));
        }

        // Extract key and value from this node - must exist for valid trunk proofs
        let (key, value) = match Self::get_key_value_from_node(&tree.node) {
            Some(key_value) => key_value,
            None => return Err(chunk_node_without_key_value(&tree.node)),
        };

        let element = Element::deserialize(&value, grove_version)?;

        // A row only reaches here under a genuine root hash, but the merk
        // root covers a tree's or reference's `value_hash`, not its bytes:
        // whether the node form binds them is version-gated (#859).
        Self::check_chunk_proof_row(&tree.node, &key, &value, &element, grove_version)?;

        elements.insert(key.clone(), element);

        // Check if this node has Hash children (making it a leaf)
        let has_hash_child = either_child_is_hash(left_is_hash, right_is_hash);

        if has_hash_child {
            // Store the node's hash and count as LeafInfo for branch queries.
            // When a branch query is made for this key, the branch proof's root hash
            // should match this node's hash.
            let node_hash = tree.hash().unwrap();

            // Extract count from TreeFeatureType if available
            // Note: KVHashCount is not included as it never reaches this code path
            let count = match &tree.node {
                Node::KVValueHashFeatureType(_, _, _, feature_type)
                | Node::KVValueHashFeatureTypeWithChildHash(_, _, _, feature_type, _) => {
                    feature_type.count()
                }
                Node::KVCount(_, _, count) => Some(*count),
                Node::KVRefValueHashCount(_, _, _, count) => Some(*count),
                // Dual-axis (PCPS) variants — count lives at the same
                // position as the single-axis count.
                Node::KVCountSum(_, _, count, _) => Some(*count),
                _ => None,
            };

            leaf_keys.insert(
                key,
                crate::query::LeafInfo {
                    hash: node_hash,
                    count,
                },
            );
        }

        // Recurse into non-Hash children
        if let Some(left) = &tree.left
            && !matches!(left.tree.node, Node::Hash(_))
        {
            Self::extract_elements_and_leaf_keys(
                &left.tree,
                elements,
                leaf_keys,
                current_depth + 1,
                max_depth,
                grove_version,
            )?;
        }
        if let Some(right) = &tree.right
            && !matches!(right.tree.node, Node::Hash(_))
        {
            Self::extract_elements_and_leaf_keys(
                &right.tree,
                elements,
                leaf_keys,
                current_depth + 1,
                max_depth,
                grove_version,
            )?;
        }

        Ok(())
    }

    /// Extract key and value from a node if it has both.
    fn get_key_value_from_node(node: &Node) -> Option<(Vec<u8>, Vec<u8>)> {
        match node {
            Node::KV(key, value)
            | Node::KVValueHash(key, value, ..)
            | Node::KVBackwardsReferencesValueHash(key, value, ..)
            | Node::KVValueHashFeatureType(key, value, ..)
            | Node::KVValueHashFeatureTypeWithChildHash(key, value, ..)
            | Node::KVCount(key, value, ..)
            | Node::KVRefValueHash(key, value, ..)
            | Node::KVRefValueHashCount(key, value, ..)
            | Node::KVSum(key, value, ..)
            | Node::KVRefValueHashSum(key, value, ..)
            | Node::KVCountSum(key, value, ..)
            | Node::KVRefValueHashCountSum(key, value, ..) => Some((key.clone(), value.clone())),
            // These nodes don't have values, only key+hash or just hash
            Node::KVDigest(..)
            | Node::KVDigestCount(..)
            | Node::KVDigestSum(..)
            | Node::KVDigestCountSum(..)
            | Node::Hash(_)
            | Node::KVHash(_)
            | Node::KVHashCount(..)
            | Node::HashWithCount(..)
            | Node::KVHashSum(..)
            | Node::HashWithSum(..)
            | Node::KVHashCountSum(..)
            | Node::HashWithCountAndSum(..) => None,
        }
    }

    /// Extract the count from a CountTree, CountSumTree, ProvableCountTree,
    /// or ProvableCountSumTree element. Looks through `NonCounted` so that a
    /// trunk proof rooted at a wrapped count tree still extracts the inner
    /// tree's count.
    fn extract_count_from_element(element: &Element) -> Option<u64> {
        match element.underlying() {
            Element::CountTree(_, count, _)
            | Element::CountSumTree(_, count, ..)
            | Element::ProvableCountTree(_, count, _)
            | Element::ProvableCountSumTree(_, count, ..)
            | Element::ProvableCountProvableSumTree(_, count, ..) => Some(*count),
            _ => None,
        }
    }

    /// Verify a serialized branch chunk proof.
    ///
    /// # Arguments
    /// * `proof` - The serialized proof bytes
    /// * `query` - The path branch chunk query
    /// * `expected_root_hash` - The expected root hash of the branch (from
    ///   parent trunk/branch proof)
    /// * `grove_version` - The GroveDB version for element deserialization
    ///
    /// # Returns
    /// `GroveBranchQueryResult` containing:
    /// - Deserialized GroveDB Elements
    /// - Leaf keys with their hashes for subsequent branch queries (if more
    ///   depth remains)
    /// - The verified branch root hash
    pub fn verify_branch_chunk_proof(
        proof: &[u8],
        query: &crate::query::PathBranchChunkQuery,
        expected_root_hash: CryptoHash,
        grove_version: &GroveVersion,
    ) -> Result<crate::query::GroveBranchQueryResult, Error> {
        // Decode the proof ops
        let mut decoder = Decoder::new(proof);
        let ops = match decode_chunk_ops(&mut decoder) {
            Ok(ops) => ops,
            Err(e) => {
                return Err(Error::CorruptedData(format!(
                    "Failed to decode branch proof: {}",
                    e
                )));
            }
        };
        if decoder.remaining_bytes() > 0 {
            return Err(Error::CorruptedData(format!(
                "Branch proof has {} unconsumed trailing bytes",
                decoder.remaining_bytes()
            )));
        }

        // Execute the proof to build the tree structure and get its root hash
        // Use collapse=false to preserve the full tree structure for element extraction
        let branch_tree = match execute_chunk_ops(&ops).unwrap() {
            Ok(branch_tree) => branch_tree,
            Err(e) => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(query.path.clone(), Query::default()),
                    format!("Failed to execute branch proof: {}", e),
                ));
            }
        };

        let branch_root_hash = branch_tree.hash().unwrap();

        // Verify the computed hash matches the expected hash from the parent proof
        if branch_root_hash != expected_root_hash {
            return Err(Error::InvalidProof(
                PathQuery::new_unsized(query.path.clone(), Query::default()),
                format!(
                    "Branch root hash mismatch: expected {}, got {}",
                    hex::encode(expected_root_hash),
                    hex::encode(branch_root_hash)
                ),
            ));
        }

        // Extract elements and leaf keys from the proof tree
        let mut elements = BTreeMap::new();
        let mut leaf_keys = BTreeMap::new();
        Self::extract_elements_and_leaf_keys(
            &branch_tree,
            &mut elements,
            &mut leaf_keys,
            0,
            query.depth as usize,
            grove_version,
        )?;

        Ok(crate::query::GroveBranchQueryResult {
            elements,
            leaf_keys,
            branch_root_hash,
            tree: branch_tree,
        })
    }
}

/// One path layer a trunk proof walked: the element bytes and value hash the
/// layer proved for the path segment, and the layer's own Merk root (the
/// child hash its parent layer commits to).
struct TrunkLayerInfo {
    value_bytes: Vec<u8>,
    expected_hash: CryptoHash,
    /// The root hash of this layer's merk tree (used as child hash for
    /// parent layer)
    layer_root_hash: CryptoHash,
}

/// The single-key query a trunk walk proves each path segment with:
/// `Query { items: vec![Key(path_segment)], ..Default::default() }`, with
/// the one-item list built by `push` (the extraction does not translate
/// that `vec![..]`).
#[allow(clippy::vec_init_then_push)]
fn single_key_query(path_segment: &[u8]) -> Query {
    let mut items = Vec::with_capacity(1);
    items.push(grovedb_merk::proofs::query::QueryItem::Key(
        path_segment.to_vec(),
    ));
    Query {
        items,
        ..Default::default()
    }
}

/// Every op of a chunk proof, decoded in order, stopping at the first
/// decoding error: `decoder.by_ref().collect::<Result<Vec<Op>, _>>()` as a
/// loop (the extraction has no model of `collect`). The decoder is left
/// where `collect` would leave it, for the trailing-bytes check.
fn decode_chunk_ops(decoder: &mut Decoder<'_>) -> Result<Vec<Op>, grovedb_merk::Error> {
    let mut ops = Vec::new();
    let mut failure = None;
    let mut done = false;
    while !done {
        match decoder.next() {
            Some(Ok(op)) => ops.push(op),
            Some(Err(e)) => {
                failure = Some(e);
                done = true;
            }
            None => done = true,
        }
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(ops),
    }
}

/// Clones of `ops` in order, each wrapped in `Ok`: the
/// `ops.iter().map(|op| Ok(op.clone()))` the chunk verifiers executed, as
/// a named iterator.
struct ClonedChunkOps<'a> {
    ops: &'a [Op],
    next: usize,
}

impl Iterator for ClonedChunkOps<'_> {
    type Item = Result<Op, grovedb_merk::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next < self.ops.len() {
            let op = self.ops[self.next].clone();
            self.next += 1;
            Some(Ok(op))
        } else {
            None
        }
    }
}

/// The node visitor of a chunk proof's execution: accepts every node (the
/// `|_| Ok(())` the chunk verifiers passed).
struct AcceptEveryNode;

impl grovedb_merk::proofs::tree::NodeVisitor for AcceptEveryNode {
    fn visit_node(&mut self, _node: &Node) -> Result<(), grovedb_merk::Error> {
        Ok(())
    }
}

/// Execute a chunk proof's ops without collapsing, with the AVL balance
/// check: `execute(ops.iter().map(|op| Ok(op.clone())), false, |_| Ok(()))`
/// (which is `execute_with_visitor(.., false, true, ..)`), without closures.
fn execute_chunk_ops(
    ops: &[Op],
) -> grovedb_costs::CostResult<grovedb_merk::proofs::tree::Tree, grovedb_merk::Error> {
    let mut cloned = ClonedChunkOps { ops, next: 0 };
    grovedb_merk::proofs::tree::execute_with_visitor(&mut cloned, false, true, &mut AcceptEveryNode)
}

/// The V0 trunk's hash chain, from the target layer up: for each layer,
/// innermost first, `combine_hash(H(value), lower) == expected`, and the
/// layer's own root becomes the next `lower`. Stops at the first mismatch;
/// `j` in the message counts layers from the bottom.
fn check_trunk_chain_v0(
    layer_infos: &[TrunkLayerInfo],
    lower_hash: CryptoHash,
    current_path: &[Vec<u8>],
) -> Result<(), Error> {
    let n = layer_infos.len();
    let mut lower_hash = lower_hash;
    let mut failure = None;
    let mut failed = false;
    let mut j = 0;
    while j < n && !failed {
        let layer_info = &layer_infos[n - 1 - j];
        match trunk_link_v0(layer_info, &lower_hash, j, current_path) {
            Ok(()) => lower_hash = layer_info.layer_root_hash,
            Err(e) => {
                failure = Some(e);
                failed = true;
            }
        }
        j += 1;
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// One link of [`check_trunk_chain_v0`].
fn trunk_link_v0(
    layer_info: &TrunkLayerInfo,
    lower_hash: &CryptoHash,
    i: usize,
    current_path: &[Vec<u8>],
) -> Result<(), Error> {
    let combined_hash = combine_hash(value_hash(&layer_info.value_bytes).value(), lower_hash)
        .value()
        .to_owned();

    if combined_hash != layer_info.expected_hash {
        return Err(Error::InvalidProof(
            PathQuery::new_unsized(current_path.to_vec(), Query::default()),
            format!(
                "Hash mismatch at layer {} from bottom: expected {}, got {}",
                i,
                hex::encode(layer_info.expected_hash),
                hex::encode(combined_hash)
            ),
        ));
    }
    Ok(())
}

/// The V1 trunk's hash chain, as [`check_trunk_chain_v0`] but checking
/// each link through [`ParentCommitment`] with the V1 message.
fn check_trunk_chain_v1(
    layer_infos: &[TrunkLayerInfo],
    lower_hash: CryptoHash,
    current_path: &[Vec<u8>],
) -> Result<(), Error> {
    let n = layer_infos.len();
    let mut current_lower_hash = lower_hash;
    let mut failure = None;
    let mut failed = false;
    let mut j = 0;
    while j < n && !failed {
        let layer_info = &layer_infos[n - 1 - j];
        match trunk_link_v1(layer_info, &current_lower_hash, j, current_path) {
            Ok(()) => current_lower_hash = layer_info.layer_root_hash,
            Err(e) => {
                failure = Some(e);
                failed = true;
            }
        }
        j += 1;
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// One link of [`check_trunk_chain_v1`].
fn trunk_link_v1(
    layer_info: &TrunkLayerInfo,
    current_lower_hash: &CryptoHash,
    i: usize,
    current_path: &[Vec<u8>],
) -> Result<(), Error> {
    let parent = ParentCommitment {
        value_bytes: &layer_info.value_bytes,
        value_hash: &layer_info.expected_hash,
    };
    match parent.check(current_lower_hash) {
        Ok(()) => Ok(()),
        Err(combined_hash) => Err(Error::InvalidProof(
            PathQuery::new_unsized(current_path.to_vec(), Query::default()),
            format!(
                "V1 trunk hash mismatch at layer {} from bottom: expected {}, got {}",
                i,
                hex::encode(layer_info.expected_hash),
                hex::encode(combined_hash)
            ),
        )),
    }
}

/// The trunk's extraction depth: the first chunk depth, or 0 if there is
/// none (`chunk_depths.first().copied().unwrap_or(0) as usize`).
fn first_chunk_depth(chunk_depths: &[u8]) -> usize {
    match chunk_depths.first() {
        Some(depth) => *depth as usize,
        None => 0,
    }
}

/// Whether a chunk-proof child is a `Node::Hash`, `None` without a child
/// (`child.as_ref().map(|c| matches!(c.tree.node, Node::Hash(_)))`).
#[allow(clippy::manual_map)] // `Option::map` with a closure is what this avoids
fn child_is_hash(child: &Option<grovedb_merk::proofs::tree::Child>) -> Option<bool> {
    match child {
        Some(c) => Some(matches!(c.tree.node, Node::Hash(_))),
        None => None,
    }
}

/// Whether either child is a `Node::Hash`
/// (`left.unwrap_or(false) || right.unwrap_or(false)`). A function of its
/// own so the extracted chunk walk does not join on the `||` of two
/// `unwrap_or`s.
fn either_child_is_hash(left: Option<bool>, right: Option<bool>) -> bool {
    matches!(left, Some(true)) || matches!(right, Some(true))
}

/// The error for a chunk-proof node with no key/value data. Error
/// construction only (the node's `Debug` text), out of line and `#[cold]`;
/// the extraction treats it as opaque.
#[cold]
fn chunk_node_without_key_value(node: &Node) -> Error {
    Error::InvalidProof(
        PathQuery::new_unsized(Vec::new(), Query::default()),
        format!(
            "Trunk proof contains node without key/value data: {:?}",
            node
        ),
    )
}

/// The `total_count` of the append-only element the single-key subset
/// query proved: the result must be exactly one row, for `key`, with a
/// present `BulkAppendTree` or `CommitmentTree` (wrappers looked through).
/// The error is the rejection message. Written without the slice pattern
/// `[(_, element_key, Some(element))]`, which the extraction cannot
/// translate.
fn bound_append_only_total_count(
    element_results: &[PathKeyOptionalElementTrio],
    key: &[u8],
) -> Result<u64, &'static str> {
    const NOT_BOUND: &str = "proof does not bind the append-only element at path/key";
    if element_results.len() != 1 {
        return Err(NOT_BOUND);
    }
    let (_, element_key, element) = &element_results[0];
    let element = match element {
        Some(element) => element,
        None => return Err(NOT_BOUND),
    };
    if element_key.as_slice() != key {
        return Err(NOT_BOUND);
    }
    match element.underlying() {
        Element::BulkAppendTree(total_count, _, _) | Element::CommitmentTree(total_count, _, _) => {
            Ok(*total_count)
        }
        _ => Err("element at path/key is not a BulkAppendTree or CommitmentTree"),
    }
}

/// The `(position, value)` entries of a verified bulk position range,
/// requiring them to be exactly `[start, end)`: ascending, contiguous,
/// complete. Each result row is checked by [`bulk_range_entry`], in
/// order, and the first failing row's error is returned; an index loop
/// whose body never returns early.
fn bulk_range_entries(
    results: &[PathKeyOptionalElementTrio],
    start: u64,
    end: u64,
    range_query: &PathQuery,
) -> Result<Vec<(u64, Vec<u8>)>, Error> {
    let mut entries = Vec::with_capacity((end - start) as usize);
    let mut expected_position = start;
    let mut failure = None;
    let mut failed = false;
    let mut i = 0;
    while i < results.len() && !failed {
        match bulk_range_entry(&results[i], expected_position, range_query) {
            Ok(entry) => {
                expected_position += 1;
                entries.push(entry);
            }
            Err(e) => {
                failure = Some(e);
                failed = true;
            }
        }
        i += 1;
    }
    if let Some(e) = failure {
        return Err(e);
    }
    if expected_position != end {
        return Err(Error::InvalidProof(
            range_query.clone(),
            format!(
                "range proof incomplete: expected positions [{}, {}), got up to {}",
                start, end, expected_position
            ),
        ));
    }
    Ok(entries)
}

/// One row of [`bulk_range_entries`]: its key must be an 8-byte
/// big-endian position, its element an item, and its position
/// `expected_position`.
fn bulk_range_entry(
    row: &PathKeyOptionalElementTrio,
    expected_position: u64,
    range_query: &PathQuery,
) -> Result<(u64, Vec<u8>), Error> {
    let (_, position_key, element) = row;
    let position = match be_u64(position_key) {
        Some(position) => position,
        None => {
            return Err(Error::InvalidProof(
                range_query.clone(),
                "range entry key is not an 8-byte big-endian position".to_string(),
            ));
        }
    };
    let value = match element {
        Some(Element::Item(value, _)) => value.clone(),
        _ => {
            return Err(Error::InvalidProof(
                range_query.clone(),
                format!("range entry at position {} is not an item", position),
            ));
        }
    };
    if position != expected_position {
        return Err(Error::InvalidProof(
            range_query.clone(),
            format!(
                "range entries not contiguous: expected position {}, got {}",
                expected_position, position
            ),
        ));
    }
    Ok((position, value))
}

/// `bytes` as a big-endian `u64` when it is exactly 8 bytes long, else
/// `None`: `<[u8; 8]>::try_from(bytes).map(u64::from_be_bytes)`, spelled
/// out because the extraction models neither.
fn be_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.len() != 8 {
        return None;
    }
    let mut value = 0u64;
    let mut i = 0;
    while i < 8 {
        value = (value << 8) | bytes[i] as u64;
        i += 1;
    }
    Some(value)
}

/// The trunk and branch chunk verifiers were reshaped for the Rust-to-Lean
/// extraction. The `ref_` functions are the previous bodies of
/// `verify_trunk_chunk_proof_v0` / `_v1`, `extract_elements_and_leaf_keys`
/// and `verify_branch_chunk_proof`, verbatim except for their names and
/// `GroveDb::` for `Self::`. The tests compare `Debug` output of the old and
/// new verifiers on real proofs (V0 and V1 envelopes) and on mutations of
/// every layer of them.
#[cfg(all(test, feature = "minimal"))]
mod chunk_proof_reference_tests {
    use grovedb_merk::proofs::execute;
    use grovedb_version::version::v1::GROVE_V1;

    use super::{chunk_decode_reference_tests::mutations, *};
    use crate::{
        query::PathBranchChunkQuery,
        tests::{common::EMPTY_PATH, make_empty_grovedb},
    };

    fn ref_verify_trunk_chunk_proof_v0(
        proof: &GroveDBProofV0,
        query: &PathTrunkChunkQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, GroveTrunkQueryResult), Error> {
        // Collect layer info as we walk down the path for later verification
        struct LayerInfo {
            value_bytes: Vec<u8>,
            expected_hash: CryptoHash,
            /// The root hash of this layer's merk tree (used as child hash for
            /// parent layer)
            layer_root_hash: CryptoHash,
        }
        let mut layer_infos: Vec<LayerInfo> = Vec::new();

        let mut current_layer = &proof.root_layer;
        let mut current_path: Vec<Vec<u8>> = Vec::new();
        let mut count: Option<u64> = None;
        let mut grovedb_root_hash: Option<CryptoHash> = None;

        // Walk through each path segment, verifying layer proofs
        for (i, path_segment) in query.path.iter().enumerate() {
            // Create a simple key query for this path segment
            let key_query = Query {
                items: vec![grovedb_merk::proofs::query::QueryItem::Key(
                    path_segment.clone(),
                )],
                ..Default::default()
            };

            // Execute the proof to verify and get the root hash
            let (layer_root_hash, result) = key_query
                .execute_proof(
                    &current_layer.merk_proof,
                    None,
                    true,
                    0, // V0 proof: allow items in value hash nodes
                )
                .unwrap()
                .map_err(|e| {
                    Error::InvalidProof(
                        PathQuery::new_unsized(current_path.clone(), key_query.clone()),
                        format!("Invalid proof at path layer {}: {}", i, e),
                    )
                })?;

            // Store the root hash of the first layer as the GroveDB root hash
            if i == 0 {
                grovedb_root_hash = Some(layer_root_hash);
            }

            // Find the element for this key in the result set
            let mut found_value_bytes: Option<Vec<u8>> = None;
            let mut found_hash: Option<CryptoHash> = None;

            for proved_key_value in &result.result_set {
                if proved_key_value.key == *path_segment {
                    found_hash = Some(proved_key_value.proof);
                    if let Some(value_bytes) = &proved_key_value.value {
                        found_value_bytes = Some(value_bytes.clone());

                        // On the last path segment, extract the count from the CountTree element
                        if i == query.path.len() - 1 {
                            let element = Element::deserialize(value_bytes, grove_version)?;
                            count = GroveDb::extract_count_from_element(&element);
                        }
                    }
                    break;
                }
            }

            let value_bytes = found_value_bytes.ok_or_else(|| {
                Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), key_query.clone()),
                    format!(
                        "Path segment {} not found in proof result",
                        hex::encode(path_segment)
                    ),
                )
            })?;

            let expected_hash = found_hash.ok_or_else(|| {
                Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), key_query.clone()),
                    format!(
                        "No hash found for path segment {}",
                        hex::encode(path_segment)
                    ),
                )
            })?;

            // Store layer info for later verification
            layer_infos.push(LayerInfo {
                value_bytes,
                expected_hash,
                layer_root_hash,
            });

            // Move to the next layer
            current_layer = current_layer
                .lower_layers
                .get(path_segment)
                .ok_or_else(|| {
                    Error::InvalidProof(
                        PathQuery::new_unsized(current_path.clone(), key_query),
                        format!(
                            "Missing lower layer for path segment {}",
                            hex::encode(path_segment)
                        ),
                    )
                })?;

            current_path.push(path_segment.clone());
        }

        // Ensure we got a count from the element
        let count = count.ok_or_else(|| {
            Error::InvalidProof(
                PathQuery::new_unsized(current_path.clone(), Query::default()),
                "Could not extract count from path - target is not a count tree element"
                    .to_string(),
            )
        })?;

        // Empty tree — return an empty result.
        // The parent layers' merk proofs already verified the hash chain integrity,
        // including the element's value_hash. The count is authenticated by the
        // parent proof. With count==0 there is no child data to verify against.
        if count == 0 {
            let grovedb_root_hash = grovedb_root_hash.ok_or_else(|| {
                Error::InvalidProof(
                    PathQuery::new_unsized(Vec::new(), Query::default()),
                    "Empty path - no root hash computed".to_string(),
                )
            })?;

            return Ok((
                grovedb_root_hash,
                GroveTrunkQueryResult {
                    elements: BTreeMap::new(),
                    leaf_keys: BTreeMap::new(),
                    chunk_depths: vec![],
                    max_tree_depth: 0,
                    tree: grovedb_merk::proofs::tree::Tree::from(Node::Hash(NULL_HASH)),
                },
            ));
        }

        let tree_depth = calculate_max_tree_depth_from_count(count);
        let chunk_depths = calculate_chunk_depths(tree_depth, query.max_depth)
            .map_err(|e| Error::CorruptedData(format!("invalid chunk depth parameters: {}", e)))?;

        // Now we're at the target layer - decode and execute the trunk proof
        let mut decoder = Decoder::new(&current_layer.merk_proof);
        let ops: Vec<Op> = decoder
            .by_ref()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::CorruptedData(format!("Failed to decode trunk proof: {}", e)))?;
        if decoder.remaining_bytes() > 0 {
            return Err(Error::CorruptedData(format!(
                "Trunk proof has {} unconsumed trailing bytes",
                decoder.remaining_bytes()
            )));
        }

        // Execute the proof to build the tree structure and get its root hash
        // Use collapse=false to preserve the full tree structure for element extraction
        let target_tree = execute(ops.iter().map(|op| Ok(op.clone())), false, |_| Ok(()))
            .unwrap()
            .map_err(|e| {
                Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    format!("Failed to execute trunk proof: {}", e),
                )
            })?;

        let mut lower_hash = target_tree.hash().unwrap();

        // Verify the cryptographic chain from trunk up through all layers
        // Walk backwards through the layer infos, verifying each step
        for (i, layer_info) in layer_infos.iter().rev().enumerate() {
            let combined_hash =
                combine_hash(value_hash(&layer_info.value_bytes).value(), &lower_hash)
                    .value()
                    .to_owned();

            if combined_hash != layer_info.expected_hash {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    format!(
                        "Hash mismatch at layer {} from bottom: expected {}, got {}",
                        i,
                        hex::encode(layer_info.expected_hash),
                        hex::encode(combined_hash)
                    ),
                ));
            }

            // For the next iteration, use this layer's merk tree root hash.
            // This is what the parent layer uses as the child hash for this subtree.
            lower_hash = layer_info.layer_root_hash;
        }

        // Extract elements and leaf keys from the proof tree
        // The max depth for the trunk is the first chunk depth
        let max_depth = chunk_depths.first().copied().unwrap_or(0) as usize;
        let mut elements = BTreeMap::new();
        let mut leaf_keys = BTreeMap::new();
        ref_extract_elements_and_leaf_keys(
            &target_tree,
            &mut elements,
            &mut leaf_keys,
            0,
            max_depth,
            grove_version,
        )?;

        let grovedb_root_hash = grovedb_root_hash.ok_or_else(|| {
            Error::InvalidProof(
                PathQuery::new_unsized(Vec::new(), Query::default()),
                "Empty path - no root hash computed".to_string(),
            )
        })?;

        let trunk_result = GroveTrunkQueryResult {
            elements,
            leaf_keys,
            chunk_depths,
            max_tree_depth: tree_depth,
            tree: target_tree,
        };

        Ok((grovedb_root_hash, trunk_result))
    }

    fn ref_verify_trunk_chunk_proof_v1(
        proof: &GroveDBProofV1,
        query: &PathTrunkChunkQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, GroveTrunkQueryResult), Error> {
        if query.path.is_empty() {
            return Err(Error::InvalidProof(
                PathQuery::new_unsized(Vec::new(), Query::default()),
                "V1 trunk: empty path - no root hash computed".to_string(),
            ));
        }

        // Collect layer info as we walk down the path for later verification
        struct LayerInfo {
            value_bytes: Vec<u8>,
            expected_hash: CryptoHash,
            /// The root hash of this layer's merk tree
            layer_root_hash: CryptoHash,
        }
        let mut layer_infos: Vec<LayerInfo> = Vec::new();

        let mut current_layer = &proof.root_layer;
        let mut current_path: Vec<Vec<u8>> = Vec::new();
        let mut count: Option<u64> = None;
        let mut grovedb_root_hash: CryptoHash = [0u8; 32];

        // Walk through each path segment, verifying layer proofs
        for (i, path_segment) in query.path.iter().enumerate() {
            let key_query = Query {
                items: vec![grovedb_merk::proofs::query::QueryItem::Key(
                    path_segment.clone(),
                )],
                ..Default::default()
            };

            // Extract merk bytes from ProofBytes::Merk
            let merk_proof_bytes = match &current_layer.merk_proof {
                ProofBytes::Merk(bytes) => bytes,
                _ => {
                    return Err(Error::InvalidProof(
                        PathQuery::new_unsized(current_path.clone(), key_query),
                        "V1 trunk proof: expected Merk proof bytes at path layer".to_string(),
                    ));
                }
            };

            let (layer_root_hash, result) = key_query
                .execute_proof(merk_proof_bytes, None, true, PROOF_VERSION_LATEST)
                .unwrap()
                .map_err(|e| {
                    Error::InvalidProof(
                        PathQuery::new_unsized(current_path.clone(), key_query.clone()),
                        format!("Invalid V1 trunk proof at path layer {}: {}", i, e),
                    )
                })?;

            if i == 0 {
                grovedb_root_hash = layer_root_hash;
            }

            let mut found_value_bytes: Option<Vec<u8>> = None;
            let mut found_hash: Option<CryptoHash> = None;

            for proved_key_value in &result.result_set {
                if proved_key_value.key == *path_segment {
                    found_hash = Some(proved_key_value.proof);
                    if let Some(value_bytes) = &proved_key_value.value {
                        found_value_bytes = Some(value_bytes.clone());

                        if i == query.path.len() - 1 {
                            let element = Element::deserialize(value_bytes, grove_version)?;
                            count = GroveDb::extract_count_from_element(&element);
                        }
                    }
                    break;
                }
            }

            let value_bytes = found_value_bytes.ok_or_else(|| {
                Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), key_query.clone()),
                    format!(
                        "V1 trunk: path segment {} not found in proof result",
                        hex::encode(path_segment)
                    ),
                )
            })?;

            let expected_hash = found_hash.ok_or_else(|| {
                Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), key_query.clone()),
                    format!(
                        "V1 trunk: no hash found for path segment {}",
                        hex::encode(path_segment)
                    ),
                )
            })?;

            layer_infos.push(LayerInfo {
                value_bytes,
                expected_hash,
                layer_root_hash,
            });

            current_layer = current_layer
                .lower_layers
                .get(path_segment)
                .ok_or_else(|| {
                    Error::InvalidProof(
                        PathQuery::new_unsized(current_path.clone(), key_query),
                        format!(
                            "V1 trunk: missing lower layer for path segment {}",
                            hex::encode(path_segment)
                        ),
                    )
                })?;

            current_path.push(path_segment.clone());
        }

        let count = count.ok_or_else(|| {
            Error::InvalidProof(
                PathQuery::new_unsized(current_path.clone(), Query::default()),
                "V1 trunk: could not extract count from path - target is not a count tree element"
                    .to_string(),
            )
        })?;

        // Extract merk bytes from the target layer
        let target_merk_bytes = match &current_layer.merk_proof {
            ProofBytes::Merk(bytes) => bytes,
            _ => {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    "V1 trunk proof: expected Merk proof bytes at target layer".to_string(),
                ));
            }
        };

        // Determine the lower hash -- always computed, even for empty trees.
        // This is the critical security fix: V0 skipped this for count==0.
        let (lower_hash, target_tree_opt) = if count == 0 {
            if !target_merk_bytes.is_empty() {
                return Err(Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    format!(
                        "V1 trunk: target layer has {} bytes but count is 0 (expected empty)",
                        target_merk_bytes.len()
                    ),
                ));
            }
            (NULL_HASH, None)
        } else {
            let tree_depth = calculate_max_tree_depth_from_count(count);
            let _chunk_depths =
                calculate_chunk_depths(tree_depth, query.max_depth).map_err(|e| {
                    Error::CorruptedData(format!("invalid chunk depth parameters: {}", e))
                })?;

            let mut decoder = Decoder::new(target_merk_bytes);
            let ops: Vec<Op> = decoder
                .by_ref()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    Error::CorruptedData(format!("Failed to decode V1 trunk proof: {}", e))
                })?;
            if decoder.remaining_bytes() > 0 {
                return Err(Error::CorruptedData(format!(
                    "V1 trunk proof has {} unconsumed trailing bytes",
                    decoder.remaining_bytes()
                )));
            }

            let target_tree = execute(ops.iter().map(|op| Ok(op.clone())), false, |_| Ok(()))
                .unwrap()
                .map_err(|e| {
                    Error::InvalidProof(
                        PathQuery::new_unsized(current_path.clone(), Query::default()),
                        format!("Failed to execute V1 trunk proof: {}", e),
                    )
                })?;

            let hash = target_tree.hash().unwrap();
            (hash, Some(target_tree))
        };

        // ALWAYS verify the cryptographic chain from trunk up through all layers.
        // This is the security fix: V0 skipped this when count==0, allowing
        // an attacker to forge the element bytes while keeping the original
        // value_hash in a KVValueHash node.
        let mut current_lower_hash = lower_hash;
        for (i, layer_info) in layer_infos.iter().rev().enumerate() {
            let parent = ParentCommitment {
                value_bytes: &layer_info.value_bytes,
                value_hash: &layer_info.expected_hash,
            };
            parent.check(&current_lower_hash).map_err(|combined_hash| {
                Error::InvalidProof(
                    PathQuery::new_unsized(current_path.clone(), Query::default()),
                    format!(
                        "V1 trunk hash mismatch at layer {} from bottom: expected {}, got {}",
                        i,
                        hex::encode(layer_info.expected_hash),
                        hex::encode(combined_hash)
                    ),
                )
            })?;

            current_lower_hash = layer_info.layer_root_hash;
        }

        // Handle empty tree early return (AFTER hash verification)
        if count == 0 {
            return Ok((
                grovedb_root_hash,
                GroveTrunkQueryResult {
                    elements: BTreeMap::new(),
                    leaf_keys: BTreeMap::new(),
                    chunk_depths: vec![],
                    max_tree_depth: 0,
                    tree: grovedb_merk::proofs::tree::Tree::from(Node::Hash(NULL_HASH)),
                },
            ));
        }

        // Non-empty tree: extract elements
        let target_tree = target_tree_opt.unwrap();
        let tree_depth = calculate_max_tree_depth_from_count(count);
        let chunk_depths = calculate_chunk_depths(tree_depth, query.max_depth)
            .map_err(|e| Error::CorruptedData(format!("invalid chunk depth parameters: {}", e)))?;

        let max_depth = chunk_depths.first().copied().unwrap_or(0) as usize;
        let mut elements = BTreeMap::new();
        let mut leaf_keys = BTreeMap::new();
        ref_extract_elements_and_leaf_keys(
            &target_tree,
            &mut elements,
            &mut leaf_keys,
            0,
            max_depth,
            grove_version,
        )?;

        let trunk_result = GroveTrunkQueryResult {
            elements,
            leaf_keys,
            chunk_depths,
            max_tree_depth: tree_depth,
            tree: target_tree,
        };

        Ok((grovedb_root_hash, trunk_result))
    }

    fn ref_extract_elements_and_leaf_keys(
        tree: &grovedb_merk::proofs::tree::Tree,
        elements: &mut BTreeMap<Vec<u8>, Element>,
        leaf_keys: &mut BTreeMap<Vec<u8>, crate::query::LeafInfo>,
        current_depth: usize,
        max_depth: usize,
        grove_version: &GroveVersion,
    ) -> Result<(), Error> {
        // Check that we haven't exceeded the max depth
        if current_depth > max_depth {
            return Err(Error::InvalidProof(
                PathQuery::new_unsized(Vec::new(), Query::default()),
                format!(
                    "Trunk proof exceeds max depth: current depth {} > max depth {}",
                    current_depth, max_depth
                ),
            ));
        }

        // Check for inconsistent depth: if one child is Hash and the other exists
        // but is not Hash, the proof has inconsistent truncation depth
        let left_is_hash = tree
            .left
            .as_ref()
            .map(|c| matches!(c.tree.node, Node::Hash(_)));
        let right_is_hash = tree
            .right
            .as_ref()
            .map(|c| matches!(c.tree.node, Node::Hash(_)));

        // If both children exist, they must both be Hash or both be non-Hash
        if let (Some(left_hash), Some(right_hash)) = (left_is_hash, right_is_hash)
            && left_hash != right_hash
        {
            return Err(Error::InvalidProof(
                PathQuery::new_unsized(Vec::new(), Query::default()),
                "Inconsistent trunk proof depth: one child is Hash while the other is not"
                    .to_string(),
            ));
        }

        // Extract key and value from this node - must exist for valid trunk proofs
        let (key, value) = GroveDb::get_key_value_from_node(&tree.node).ok_or_else(|| {
            Error::InvalidProof(
                PathQuery::new_unsized(Vec::new(), Query::default()),
                format!(
                    "Trunk proof contains node without key/value data: {:?}",
                    tree.node
                ),
            )
        })?;

        let element = Element::deserialize(&value, grove_version)?;

        // A row only reaches here under a genuine root hash, but the merk
        // root covers a tree's or reference's `value_hash`, not its bytes:
        // whether the node form binds them is version-gated (#859).
        GroveDb::check_chunk_proof_row(&tree.node, &key, &value, &element, grove_version)?;

        elements.insert(key.clone(), element);

        // Check if this node has Hash children (making it a leaf)
        let has_hash_child = left_is_hash.unwrap_or(false) || right_is_hash.unwrap_or(false);

        if has_hash_child {
            // Store the node's hash and count as LeafInfo for branch queries.
            // When a branch query is made for this key, the branch proof's root hash
            // should match this node's hash.
            let node_hash = tree.hash().unwrap();

            // Extract count from TreeFeatureType if available
            // Note: KVHashCount is not included as it never reaches this code path
            let count = match &tree.node {
                Node::KVValueHashFeatureType(_, _, _, feature_type)
                | Node::KVValueHashFeatureTypeWithChildHash(_, _, _, feature_type, _) => {
                    feature_type.count()
                }
                Node::KVCount(_, _, count) => Some(*count),
                Node::KVRefValueHashCount(_, _, _, count) => Some(*count),
                // Dual-axis (PCPS) variants — count lives at the same
                // position as the single-axis count.
                Node::KVCountSum(_, _, count, _) => Some(*count),
                _ => None,
            };

            leaf_keys.insert(
                key,
                crate::query::LeafInfo {
                    hash: node_hash,
                    count,
                },
            );
        }

        // Recurse into non-Hash children
        if let Some(left) = &tree.left
            && !matches!(left.tree.node, Node::Hash(_))
        {
            ref_extract_elements_and_leaf_keys(
                &left.tree,
                elements,
                leaf_keys,
                current_depth + 1,
                max_depth,
                grove_version,
            )?;
        }
        if let Some(right) = &tree.right
            && !matches!(right.tree.node, Node::Hash(_))
        {
            ref_extract_elements_and_leaf_keys(
                &right.tree,
                elements,
                leaf_keys,
                current_depth + 1,
                max_depth,
                grove_version,
            )?;
        }

        Ok(())
    }

    fn ref_verify_branch_chunk_proof(
        proof: &[u8],
        query: &crate::query::PathBranchChunkQuery,
        expected_root_hash: CryptoHash,
        grove_version: &GroveVersion,
    ) -> Result<crate::query::GroveBranchQueryResult, Error> {
        // Decode the proof ops
        let mut decoder = Decoder::new(proof);
        let ops: Vec<Op> = decoder
            .by_ref()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Error::CorruptedData(format!("Failed to decode branch proof: {}", e)))?;
        if decoder.remaining_bytes() > 0 {
            return Err(Error::CorruptedData(format!(
                "Branch proof has {} unconsumed trailing bytes",
                decoder.remaining_bytes()
            )));
        }

        // Execute the proof to build the tree structure and get its root hash
        // Use collapse=false to preserve the full tree structure for element extraction
        let branch_tree = execute(ops.iter().map(|op| Ok(op.clone())), false, |_| Ok(()))
            .unwrap()
            .map_err(|e| {
                Error::InvalidProof(
                    PathQuery::new_unsized(query.path.clone(), Query::default()),
                    format!("Failed to execute branch proof: {}", e),
                )
            })?;

        let branch_root_hash = branch_tree.hash().unwrap();

        // Verify the computed hash matches the expected hash from the parent proof
        if branch_root_hash != expected_root_hash {
            return Err(Error::InvalidProof(
                PathQuery::new_unsized(query.path.clone(), Query::default()),
                format!(
                    "Branch root hash mismatch: expected {}, got {}",
                    hex::encode(expected_root_hash),
                    hex::encode(branch_root_hash)
                ),
            ));
        }

        // Extract elements and leaf keys from the proof tree
        let mut elements = BTreeMap::new();
        let mut leaf_keys = BTreeMap::new();
        ref_extract_elements_and_leaf_keys(
            &branch_tree,
            &mut elements,
            &mut leaf_keys,
            0,
            query.depth as usize,
            grove_version,
        )?;

        Ok(crate::query::GroveBranchQueryResult {
            elements,
            leaf_keys,
            branch_root_hash,
            tree: branch_tree,
        })
    }

    fn ref_verify_trunk_chunk_proof(
        proof: &[u8],
        query: &PathTrunkChunkQuery,
        grove_version: &GroveVersion,
    ) -> Result<(CryptoHash, GroveTrunkQueryResult), Error> {
        let grovedb_proof = super::super::decode_grovedb_proof_canonical(proof)?;
        match grovedb_proof {
            GroveDBProof::V0(proof_v0) => {
                ref_verify_trunk_chunk_proof_v0(&proof_v0, query, grove_version)
            }
            GroveDBProof::V1(proof_v1) => {
                ref_verify_trunk_chunk_proof_v1(&proof_v1, query, grove_version)
            }
        }
    }

    fn config() -> impl bincode::config::Config {
        bincode::config::standard()
            .with_big_endian()
            .with_no_limit()
    }

    fn compare_trunk(proof: &[u8], query: &PathTrunkChunkQuery, grove_version: &GroveVersion) {
        let got = GroveDb::verify_trunk_chunk_proof(proof, query, grove_version);
        let want = ref_verify_trunk_chunk_proof(proof, query, grove_version);
        assert_eq!(format!("{got:?}"), format!("{want:?}"), "{query:?}");
    }

    fn v1_layer_at_mut<'a>(
        layer: &'a mut LayerProof,
        path: &[Vec<u8>],
    ) -> Option<&'a mut LayerProof> {
        match path.split_first() {
            None => Some(layer),
            Some((first, rest)) => match layer.lower_layers.get_mut(first) {
                Some(lower) => v1_layer_at_mut(lower, rest),
                None => None,
            },
        }
    }

    fn v0_layer_at_mut<'a>(
        layer: &'a mut MerkOnlyLayerProof,
        path: &[Vec<u8>],
    ) -> Option<&'a mut MerkOnlyLayerProof> {
        match path.split_first() {
            None => Some(layer),
            Some((first, rest)) => match layer.lower_layers.get_mut(first) {
                Some(lower) => v0_layer_at_mut(lower, rest),
                None => None,
            },
        }
    }

    fn to_merk_only(layer: &LayerProof) -> MerkOnlyLayerProof {
        let merk_proof = match &layer.merk_proof {
            ProofBytes::Merk(bytes) => bytes.clone(),
            _ => Vec::new(),
        };
        let mut lower_layers = BTreeMap::new();
        for (k, v) in &layer.lower_layers {
            lower_layers.insert(k.clone(), to_merk_only(v));
        }
        MerkOnlyLayerProof {
            merk_proof,
            lower_layers,
        }
    }

    fn decode(proof: &[u8]) -> GroveDBProof {
        bincode::decode_from_slice(proof, config()).unwrap().0
    }

    /// The merk bytes of the layer at `prefix`, if the proof has one.
    fn layer_bytes(proof: &[u8], prefix: &[Vec<u8>]) -> Option<Vec<u8>> {
        match decode(proof) {
            GroveDBProof::V1(mut v1) => match v1_layer_at_mut(&mut v1.root_layer, prefix) {
                Some(layer) => match &layer.merk_proof {
                    ProofBytes::Merk(bytes) => Some(bytes.clone()),
                    _ => None,
                },
                None => None,
            },
            GroveDBProof::V0(mut v0) => {
                v0_layer_at_mut(&mut v0.root_layer, prefix).map(|layer| layer.merk_proof.clone())
            }
        }
    }

    /// `proof` with the layer at `prefix` edited: its merk bytes replaced
    /// (`Some`), or its lower layers dropped (`None`). `as_mmr` swaps a V1
    /// layer's bytes to a non-Merk variant instead.
    fn edited(proof: &[u8], prefix: &[Vec<u8>], bytes: Option<Vec<u8>>, as_mmr: bool) -> Vec<u8> {
        let mut decoded = decode(proof);
        match &mut decoded {
            GroveDBProof::V1(v1) => {
                let layer = v1_layer_at_mut(&mut v1.root_layer, prefix).unwrap();
                match bytes {
                    Some(b) if as_mmr => layer.merk_proof = ProofBytes::MMR(b),
                    Some(b) => layer.merk_proof = ProofBytes::Merk(b),
                    None => layer.lower_layers.clear(),
                }
            }
            GroveDBProof::V0(v0) => {
                let layer = v0_layer_at_mut(&mut v0.root_layer, prefix).unwrap();
                match bytes {
                    Some(b) => layer.merk_proof = b,
                    None => layer.lower_layers.clear(),
                }
            }
        }
        bincode::encode_to_vec(&decoded, config()).unwrap()
    }

    /// `proof` as a V0 envelope, if it is V1.
    fn as_v0(proof: &[u8]) -> Option<Vec<u8>> {
        match decode(proof) {
            GroveDBProof::V1(v1) => {
                let v0 = GroveDBProof::V0(GroveDBProofV0 {
                    root_layer: to_merk_only(&v1.root_layer),
                    prove_options: ProveOptions::default(),
                });
                Some(bincode::encode_to_vec(&v0, config()).unwrap())
            }
            GroveDBProof::V0(_) => None,
        }
    }

    /// The verifier's view of `proof` and of every mutation of each layer
    /// along `query.path` (and the target), plus a dropped lower layer, a
    /// non-Merk layer, and the V0 conversion of each V1 variant.
    fn compare_trunk_family(
        proof: &[u8],
        query: &PathTrunkChunkQuery,
        grove_version: &GroveVersion,
    ) {
        let mut variants = vec![proof.to_vec()];
        for depth in 0..=query.path.len() {
            let prefix = &query.path[..depth];
            let Some(bytes) = layer_bytes(proof, prefix) else {
                continue;
            };
            for m in mutations(&bytes) {
                variants.push(edited(proof, prefix, Some(m), false));
            }
            variants.push(edited(proof, prefix, Some(bytes.clone()), true));
            variants.push(edited(proof, prefix, None, false));
        }
        let converted: Vec<Vec<u8>> = variants.iter().filter_map(|v| as_v0(v)).collect();
        variants.extend(converted);
        for v in variants {
            compare_trunk(&v, query, grove_version);
        }
    }

    /// Count trees of several sizes and kinds under one- and two-segment
    /// paths, proved at V1 (latest) and V0 (GROVE_V1) prover versions.
    #[test]
    fn trunk_and_branch_verifiers_match_reference() {
        let latest = GroveVersion::latest();
        let db = make_empty_grovedb();
        db.insert(
            EMPTY_PATH,
            b"root",
            Element::empty_tree(),
            None,
            None,
            latest,
        )
        .unwrap()
        .unwrap();
        let trees: [(&[u8], Element, u32); 5] = [
            (b"c0", Element::empty_count_tree(), 0),
            (b"c1", Element::empty_count_tree(), 1),
            (b"c7", Element::empty_provable_count_tree(), 7),
            (b"cs", Element::empty_count_sum_tree(), 40),
            (b"it", Element::empty_tree(), 3),
        ];
        let mut queries = Vec::new();
        for (key, element, n) in trees {
            for parent in [vec![], vec![b"root".to_vec()]] {
                let parent_refs: Vec<&[u8]> = parent.iter().map(|p| p.as_slice()).collect();
                db.insert(
                    parent_refs.as_slice(),
                    key,
                    element.clone(),
                    None,
                    None,
                    latest,
                )
                .unwrap()
                .unwrap();
                let mut path = parent.clone();
                path.push(key.to_vec());
                let path_refs: Vec<&[u8]> = path.iter().map(|p| p.as_slice()).collect();
                for i in 0..n {
                    db.insert(
                        path_refs.as_slice(),
                        &i.to_be_bytes(),
                        if key == b"cs" {
                            Element::new_sum_item(i as i64 + 1)
                        } else {
                            Element::new_item(i.to_be_bytes().to_vec())
                        },
                        None,
                        None,
                        latest,
                    )
                    .unwrap()
                    .unwrap();
                }
                for max_depth in [2u8, 3, 8] {
                    queries.push(PathTrunkChunkQuery::new(path.clone(), max_depth));
                }
            }
        }
        let mut branch_cases = 0;
        for query in &queries {
            for grove_version in [latest, &GROVE_V1] {
                let Ok(proof) = db.prove_trunk_chunk(query, grove_version).unwrap() else {
                    continue;
                };
                compare_trunk_family(&proof, query, latest);
                // Branches off the trunk's leaf keys, with the right and a
                // wrong expected hash, and mutated proofs.
                let Ok((_, trunk)) = GroveDb::verify_trunk_chunk_proof(&proof, query, latest)
                else {
                    continue;
                };
                for (leaf_key, leaf_info) in trunk.leaf_keys.iter().take(3) {
                    for depth in [1u8, 2, 4] {
                        let branch_query =
                            PathBranchChunkQuery::new(query.path.clone(), leaf_key.clone(), depth);
                        let Ok(branch_proof) =
                            db.prove_branch_chunk(&branch_query, latest).unwrap()
                        else {
                            continue;
                        };
                        let mut proofs = mutations(&branch_proof);
                        proofs.push(branch_proof);
                        for p in &proofs {
                            for expected in [leaf_info.hash, [9; 32]] {
                                let got = GroveDb::verify_branch_chunk_proof(
                                    p,
                                    &branch_query,
                                    expected,
                                    latest,
                                );
                                let want = ref_verify_branch_chunk_proof(
                                    p,
                                    &branch_query,
                                    expected,
                                    latest,
                                );
                                assert_eq!(format!("{got:?}"), format!("{want:?}"));
                                branch_cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(branch_cases > 100, "{branch_cases}");
    }
}

/// `decode_chunk_ops` replaces `decoder.by_ref().collect()` in the chunk
/// verifiers; it must return the same ops or error and leave the decoder
/// in the same place.
#[cfg(test)]
mod chunk_decode_reference_tests {
    use super::*;

    /// Byte-level mutations of one merk proof: flips at a stride, a
    /// truncation, an appended byte, and the empty proof.
    pub(super) fn mutations(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let stride = (bytes.len() / 40).max(1);
        let mut i = 0;
        while i < bytes.len() {
            let mut m = bytes.to_vec();
            m[i] ^= 0x01;
            out.push(m);
            let mut m = bytes.to_vec();
            m[i] = 0xff;
            out.push(m);
            i += stride;
        }
        out.push(bytes[..bytes.len() / 2].to_vec());
        let mut m = bytes.to_vec();
        m.push(0);
        out.push(m);
        out.push(Vec::new());
        out
    }

    #[test]
    fn decode_chunk_ops_matches_collect() {
        let mut ops_bytes = Vec::new();
        grovedb_merk::proofs::encoding::encode_into(
            [
                Op::Push(Node::Hash([1; 32])),
                Op::Push(Node::KV(b"k".to_vec(), b"v".to_vec())),
                Op::Parent,
            ]
            .iter(),
            &mut ops_bytes,
        );
        for bytes in mutations(&ops_bytes).into_iter().chain([ops_bytes.clone()]) {
            let mut a = Decoder::new(&bytes);
            let got = decode_chunk_ops(&mut a);
            let mut b = Decoder::new(&bytes);
            let want: Result<Vec<Op>, _> = b.by_ref().collect();
            assert_eq!(format!("{got:?}"), format!("{want:?}"));
            assert_eq!(a.remaining_bytes(), b.remaining_bytes());
        }
    }
}

/// The bulk position-range row checks were lifted out of
/// `verify_bulk_position_range_proof` for the Rust-to-Lean extraction.
/// `ref_bulk_range_entries` is the previous loop and completeness check,
/// verbatim, wrapped in a function of its inputs.
#[cfg(test)]
mod bulk_range_reference_tests {
    use super::*;

    fn ref_bulk_range_entries(
        results: Vec<PathKeyOptionalElementTrio>,
        start: u64,
        end: u64,
        range_query: PathQuery,
    ) -> Result<Vec<(u64, Vec<u8>)>, Error> {
        let mut entries = Vec::with_capacity((end - start) as usize);
        let mut expected_position = start;
        for (_, position_key, element) in results {
            let position_bytes: [u8; 8] = position_key.as_slice().try_into().map_err(|_| {
                Error::InvalidProof(
                    range_query.clone(),
                    "range entry key is not an 8-byte big-endian position".to_string(),
                )
            })?;
            let position = u64::from_be_bytes(position_bytes);
            let value = match element {
                Some(Element::Item(value, _)) => value,
                _ => {
                    return Err(Error::InvalidProof(
                        range_query,
                        format!("range entry at position {} is not an item", position),
                    ));
                }
            };
            if position != expected_position {
                return Err(Error::InvalidProof(
                    range_query,
                    format!(
                        "range entries not contiguous: expected position {}, got {}",
                        expected_position, position
                    ),
                ));
            }
            expected_position += 1;
            entries.push((position, value));
        }
        if expected_position != end {
            return Err(Error::InvalidProof(
                range_query,
                format!(
                    "range proof incomplete: expected positions [{}, {}), got up to {}",
                    start, end, expected_position
                ),
            ));
        }
        Ok(entries)
    }

    fn row(key: Vec<u8>, element: Option<Element>) -> PathKeyOptionalElementTrio {
        (vec![b"p".to_vec()], key, element)
    }

    /// Rows over a few start positions: valid runs, gaps, repeats, short
    /// and long keys, and non-item elements.
    fn row_lists(start: u64) -> Vec<Vec<PathKeyOptionalElementTrio>> {
        let pos = |p: u64| p.to_be_bytes().to_vec();
        let item = |v: u8| Some(Element::new_item(vec![v]));
        let mut lists = Vec::new();
        for n in 0..4u64 {
            lists.push(
                (0..n)
                    .map(|i| row(pos(start.wrapping_add(i)), item(i as u8)))
                    .collect(),
            );
        }
        lists.push(vec![row(pos(start.wrapping_add(1)), item(0))]);
        lists.push(vec![row(pos(start), item(0)), row(pos(start), item(1))]);
        lists.push(vec![row(pos(start)[1..].to_vec(), item(0))]);
        lists.push(vec![row([pos(start), vec![0]].concat(), item(0))]);
        lists.push(vec![row(pos(start), None)]);
        lists.push(vec![row(pos(start), Some(Element::empty_tree()))]);
        lists.push(vec![row(pos(start), item(0)), row(vec![1, 2], None)]);
        lists.push(vec![
            row(pos(start), item(0)),
            row(pos(start.wrapping_add(5)), None),
        ]);
        lists
    }

    #[test]
    fn bulk_range_entries_match_reference() {
        let range_query = PathQuery::new_single_key(vec![b"p".to_vec()], b"k".to_vec());
        for start in [0u64, 7, u64::MAX - 3] {
            for end in [start, start + 1, start + 2, start + 3] {
                for rows in row_lists(start) {
                    let got = bulk_range_entries(&rows, start, end, &range_query);
                    let want =
                        ref_bulk_range_entries(rows.clone(), start, end, range_query.clone());
                    assert_eq!(
                        format!("{got:?}"),
                        format!("{want:?}"),
                        "{start} {end} {rows:?}"
                    );
                }
            }
        }
    }

    /// The previous `total_count` match, verbatim, with its two `return`
    /// statements yielding the message instead.
    fn ref_total_count(
        element_results: Vec<PathKeyOptionalElementTrio>,
        key: &[u8],
    ) -> Result<u64, &'static str> {
        let total_count = match element_results.as_slice() {
            [(_, element_key, Some(element))] if element_key.as_slice() == key => {
                match element.underlying() {
                    Element::BulkAppendTree(total_count, _, _)
                    | Element::CommitmentTree(total_count, _, _) => *total_count,
                    _ => {
                        return Err(
                            "element at path/key is not a BulkAppendTree or CommitmentTree",
                        );
                    }
                }
            }
            _ => {
                return Err("proof does not bind the append-only element at path/key");
            }
        };
        Ok(total_count)
    }

    #[test]
    fn bound_total_count_matches_reference() {
        let elements = [
            None,
            Some(Element::BulkAppendTree(5, 2, None)),
            Some(Element::CommitmentTree(7, 2, None)),
            Some(Element::new_item(vec![1])),
            Some(Element::empty_tree()),
            Some(Element::NonCounted(Box::new(Element::BulkAppendTree(
                9, 2, None,
            )))),
        ];
        let keys: [&[u8]; 2] = [b"k", b"j"];
        let mut lists: Vec<Vec<PathKeyOptionalElementTrio>> = vec![vec![]];
        for e in &elements {
            for k in keys {
                lists.push(vec![(vec![], k.to_vec(), e.clone())]);
                lists.push(vec![
                    (vec![], k.to_vec(), e.clone()),
                    (vec![], b"k".to_vec(), e.clone()),
                ]);
            }
        }
        for rows in lists {
            assert_eq!(
                bound_append_only_total_count(&rows, b"k"),
                ref_total_count(rows.clone(), b"k"),
                "{rows:?}"
            );
        }
    }

    #[test]
    fn be_u64_matches_from_be_bytes() {
        let bytes: Vec<u8> = (1..=10).collect();
        for len in 0..=bytes.len() {
            for offset in 0..=(bytes.len() - len) {
                let slice = &bytes[offset..offset + len];
                let want: Option<u64> = <[u8; 8]>::try_from(slice).ok().map(u64::from_be_bytes);
                assert_eq!(be_u64(slice), want, "{slice:?}");
            }
        }
        assert_eq!(be_u64(&[0xff; 8]), Some(u64::MAX));
    }
}

#[cfg(test)]
mod axis_helper_tests {
    use super::*;

    /// Every axes list of length 0..=4 over three tags and three secondary
    /// key shapes (absent, empty, non-empty).
    fn every_axes_list() -> Vec<Vec<(u8, Option<Vec<u8>>)>> {
        let entries: Vec<(u8, Option<Vec<u8>>)> = [0u8, 1, 2]
            .into_iter()
            .flat_map(|t| [(t, None), (t, Some(Vec::new())), (t, Some(vec![7]))])
            .collect();
        let mut lists = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..4 {
            let mut next = Vec::new();
            for list in &frontier {
                for entry in &entries {
                    let mut extended: Vec<(u8, Option<Vec<u8>>)> = list.clone();
                    extended.push(entry.clone());
                    next.push(extended);
                }
            }
            lists.extend(next.iter().cloned());
            frontier = next;
        }
        lists
    }

    #[test]
    fn axis_helpers_match_iterator_forms() {
        let lists = every_axes_list();
        assert_eq!(lists.len(), 1 + 9 + 81 + 729 + 6561);
        for axes in &lists {
            assert_eq!(
                all_axes_unset(axes),
                axes.iter().all(|(_, sk)| sk.is_none()),
                "{axes:?}"
            );
            let old: Vec<(u8, CryptoHash)> = axes.iter().map(|(t, _)| (*t, NULL_HASH)).collect();
            assert_eq!(null_axis_hashes(axes), old, "{axes:?}");
        }
    }
}

#[cfg(test)]
mod lower_layer_helper_tests {
    use super::*;

    /// The succinctness check as `verify_layer_proof_v1` wrote it before
    /// `first_unrequested_lower_layer`: the message of the error it
    /// returned, if any.
    #[allow(clippy::manual_find)]
    fn old_succinctness(
        lower_layers: &BTreeMap<Vec<u8>, LayerProof>,
        verified_keys: &BTreeSet<Vec<u8>>,
    ) -> Option<String> {
        for key in lower_layers.keys() {
            if !verified_keys.contains(key) {
                return Some(format!(
                    "V1 proof contains extra lower layer for key {} not required by query",
                    hex::encode(key),
                ));
            }
        }
        None
    }

    /// The same check through the helper, as `verify_layer_proof_v1` now
    /// writes it.
    fn new_succinctness(
        lower_layers: &BTreeMap<Vec<u8>, LayerProof>,
        verified_keys: &BTreeSet<Vec<u8>>,
    ) -> Option<String> {
        if let Some(key) = first_unrequested_lower_layer(lower_layers, verified_keys) {
            return Some(format!(
                "V1 proof contains extra lower layer for key {} not required by query",
                hex::encode(key),
            ));
        }
        None
    }

    fn layer() -> LayerProof {
        LayerProof {
            merk_proof: ProofBytes::Merk(Vec::new()),
            lower_layers: BTreeMap::new(),
        }
    }

    /// Every subset of a six-key universe as the lower layers, against every
    /// subset of that universe plus one key outside it as the verified set.
    #[test]
    fn first_unrequested_lower_layer_matches_keys_loop() {
        let universe: Vec<Vec<u8>> = vec![
            Vec::new(),
            vec![0],
            vec![0, 0],
            vec![1],
            vec![1, 2],
            vec![0xff],
        ];
        let mut verified_universe = universe.clone();
        verified_universe.push(vec![9, 9]);
        let mut checked = 0usize;
        let mut found = 0usize;
        for layer_mask in 0u32..(1 << universe.len()) {
            let lower_layers: BTreeMap<Vec<u8>, LayerProof> = (0..universe.len())
                .filter(|i| layer_mask & (1 << i) != 0)
                .map(|i| (universe[i].clone(), layer()))
                .collect();
            for verified_mask in 0u32..(1 << verified_universe.len()) {
                let verified_keys: BTreeSet<Vec<u8>> = (0..verified_universe.len())
                    .filter(|i| verified_mask & (1 << i) != 0)
                    .map(|i| verified_universe[i].clone())
                    .collect();
                assert_eq!(
                    new_succinctness(&lower_layers, &verified_keys),
                    old_succinctness(&lower_layers, &verified_keys),
                    "{:?} {verified_keys:?}",
                    lower_layers.keys().collect::<Vec<_>>()
                );
                // Spec: the smallest lower-layer key that was not verified,
                // borrowed from the map itself.
                let found_key = first_unrequested_lower_layer(&lower_layers, &verified_keys);
                let smallest = lower_layers
                    .keys()
                    .filter(|k| !verified_keys.contains(*k))
                    .min();
                assert_eq!(
                    found_key,
                    smallest,
                    "{:?} {verified_keys:?}",
                    lower_layers.keys().collect::<Vec<_>>()
                );
                if let Some(key) = found_key {
                    let (map_key, _) = lower_layers.get_key_value(key).unwrap();
                    assert!(std::ptr::eq(key, map_key));
                    found += 1;
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 64 * 128);
        assert!(found > 0 && found < checked);
    }
}

#[cfg(test)]
mod row_sink_tests {
    use grovedb_version::version::GROVE_VERSIONS;

    use super::*;
    use crate::reference_path::ReferencePathType;

    fn row(value: Option<Vec<u8>>) -> ProvedPathKeyOptionalValue {
        ProvedPathKeyOptionalValue {
            path: vec![b"a".to_vec(), Vec::new()],
            key: b"k".to_vec(),
            value,
            proof: [7u8; 32],
        }
    }

    /// Absent values, valid encodings of a spread of element kinds, their
    /// truncations and trailing-byte forms, and a few garbage inputs.
    fn values() -> Vec<Option<Vec<u8>>> {
        let grove_version = GroveVersion::latest();
        let elements = [
            Element::new_item(b"v".to_vec()),
            Element::new_item_with_flags(b"v".to_vec(), Some(vec![1, 2])),
            Element::new_sum_item(-3),
            Element::empty_tree(),
            Element::new_tree(Some(b"r".to_vec())),
            Element::new_sum_tree(None),
            Element::new_reference(ReferencePathType::SiblingReference(b"s".to_vec())),
            Element::new_non_counted(Element::new_item(b"n".to_vec())).unwrap(),
            Element::new_not_summed(Element::new_sum_tree(None)).unwrap(),
        ];
        let mut values = vec![None, Some(Vec::new()), Some(vec![0xff; 9])];
        for element in elements {
            let bytes = element.serialize(grove_version).unwrap();
            for len in 0..bytes.len() {
                values.push(Some(bytes[..len].to_vec()));
            }
            let mut trailing = bytes.clone();
            trailing.push(0);
            values.push(Some(trailing));
            values.push(Some(bytes));
        }
        values
    }

    #[test]
    fn push_row_matches_try_into_versioned() {
        for grove_version in GROVE_VERSIONS {
            for value in values() {
                let old: Result<PathKeyOptionalElementTrio, Error> =
                    row(value.clone()).try_into_versioned(grove_version);
                let mut trios: Vec<PathKeyOptionalElementTrio> = Vec::new();
                let new = trios.push_row(row(value.clone()), grove_version);
                assert_eq!(
                    format!("{:?}", new.map(|()| trios.clone())),
                    format!("{:?}", old.map(|trio| vec![trio])),
                    "{value:?}"
                );

                // The raw conversion fails with a merk error; the walk's
                // `?` converted it with `From<grovedb_merk::Error>`.
                let old: Result<ProvedPathKeyValue, grovedb_merk::Error> =
                    row(value.clone()).try_into_versioned(grove_version);
                let old = old.map_err(Error::from);
                let mut raw: Vec<ProvedPathKeyValue> = Vec::new();
                let new = raw.push_row(row(value.clone()), grove_version);
                assert_eq!(
                    format!("{:?}", new.map(|()| raw)),
                    format!("{:?}", old.map(|kv| vec![kv])),
                    "{value:?}"
                );
            }
        }
    }
}
