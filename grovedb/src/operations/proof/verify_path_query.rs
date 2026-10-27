//! The unified verify entry point: one function that verifies a proof
//! for every provable [`PathQuery`] shape.
//!
//! [`GroveDb::verify_path_query`] classifies the query once
//! ([`PathQuery::classify`]) and routes it to the verifier that serves
//! that shape, returning a [`VerifiedPathQuery`] whose variant mirrors
//! the shape — the proved counterpart of `run_path_query`.
//!
//! Axis shapes verify **GroveDBProof V1** envelopes carrying
//! [`ProofBytes::IndexedTreeAxisDescent`](crate::operations::proof::ProofBytes)
//! layers, gated on `proof.axis_descent_in_v1_envelope` (GROVE_V4+).
//! Sum-budget shapes verify V1 envelopes carrying
//! [`ProofBytes::SumBudgetWindow`](crate::operations::proof::ProofBytes)
//! layers, gated on `proof.sum_budget_in_v1_envelope` (GROVE_V4+), with
//! the stop condition attested by the verifier's fold replay.
//! Key-selection and aggregate shapes route to the existing verifiers
//! unchanged.

use std::collections::BTreeMap;

use grovedb_merk::{
    proofs::query::{query_item::QueryItem, AxisQuery, AxisTraversal},
    CryptoHash,
};
use grovedb_version::version::GroveVersion;

use crate::{
    operations::proof::{
        decode_grovedb_proof_canonical,
        indexed_axis::AxisEntries,
        verify::{AxisWalkOutcome, AxisWalkResult},
    },
    query::{validated::ValidatedPathQuery, AggregateKind, PathQueryShape},
    query_result_type::PathKeyOptionalElementTrio,
    Error, GroveDb, PathQuery,
};

pub use crate::operations::proof::verify::SumBudgetStop;

/// The verified answer to any provable [`PathQuery`] shape — one
/// variant per shape family. Every variant carries the reconstructed
/// GroveDB root hash; compare it against the root you trust —
/// verification alone proves internal consistency, not that the proof
/// is about the state you meant.
#[derive(Debug)]
pub enum VerifiedPathQuery {
    /// Key-selection shapes: the regular trio result set.
    Elements {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// The proved `(path, key, element)` rows.
        elements: Vec<PathKeyOptionalElementTrio>,
    },
    /// Leaf `AggregateCountOnRange`.
    AggregateCount {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// The proved count.
        count: u64,
    },
    /// Leaf `AggregateSumOnRange`.
    AggregateSum {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// The proved signed sum.
        sum: i64,
    },
    /// Leaf `AggregateCountAndSumOnRange`.
    AggregateCountAndSum {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// The proved count.
        count: u64,
        /// The proved signed sum.
        sum: i64,
    },
    /// Carrier aggregates: one value per matched outer key.
    AggregatePerKey {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// Per outer key: the proved `(count, sum)` — count-only
        /// carriers report `sum = None`, sum-only carriers report
        /// `count = None`.
        per_key: Vec<(Vec<u8>, Option<u64>, Option<i64>)>,
    },
    /// Single-path axis read (`TopK` / `Bounded`): the proved entries.
    AxisEntries {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// The proved entries, in walk order.
        entries: AxisEntries,
        /// Count-commitment-attested skip for paginated traversals;
        /// `None` for bounded ones.
        skipped: Option<u64>,
    },
    /// Branched axis read: per branch key, in query order, the proved
    /// entries — or `None` for a branch key whose absence the
    /// branching-level Merk proof authenticates.
    BranchedAxisEntries {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// `(branch_key, entries)` per branch, `None` = proven absent.
        branches: Vec<(Vec<u8>, Option<AxisEntries>)>,
    },
    /// `RankOfKey`: the attested 0-based rank of the queried key in the
    /// directional walk.
    AxisRank {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// The attested rank.
        rank: u64,
    },
    /// `AggregateOverValueRange`: the attested aggregate over the entries the
    /// value range selected.
    AxisAggregate {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// The attested aggregate, whose meaning follows the queried
        /// FOLD, not the axis:
        ///
        /// * `AggregateFold::Population` — HOW MANY entries the value
        ///   range selected (`>= 0`, each entry contributing 1).
        /// * `AggregateFold::Total` — the signed sum of the selected
        ///   entries' axis values.
        ///
        /// The query carries the fold, so the caller already knows
        /// which reading applies.
        value: i128,
    },
    /// Sum-budget read: the proved window's matched sum items, their
    /// net total, and the replay-attested stop condition.
    SumBudget {
        /// Reconstructed GroveDB root hash.
        root_hash: CryptoHash,
        /// The matched `(key, value)` pairs, in walk order.
        matches: Vec<(Vec<u8>, i64)>,
        /// Net total of the matched values (saturating).
        total: i64,
        /// Why the walk stopped, attested by the fold replay.
        stop: SumBudgetStop,
    },
}

impl VerifiedPathQuery {
    /// The reconstructed GroveDB root hash, whichever shape this is.
    pub fn root_hash(&self) -> &CryptoHash {
        match self {
            VerifiedPathQuery::Elements { root_hash, .. }
            | VerifiedPathQuery::AggregateCount { root_hash, .. }
            | VerifiedPathQuery::AggregateSum { root_hash, .. }
            | VerifiedPathQuery::AggregateCountAndSum { root_hash, .. }
            | VerifiedPathQuery::AggregatePerKey { root_hash, .. }
            | VerifiedPathQuery::AxisEntries { root_hash, .. }
            | VerifiedPathQuery::BranchedAxisEntries { root_hash, .. }
            | VerifiedPathQuery::AxisRank { root_hash, .. }
            | VerifiedPathQuery::AxisAggregate { root_hash, .. }
            | VerifiedPathQuery::SumBudget { root_hash, .. } => root_hash,
        }
    }
}

impl GroveDb {
    /// Verify `proof` against `path_query`, whatever shape the query
    /// is, returning the shape's typed verified answer. The unified
    /// counterpart of the per-shape `verify_*` entry points and the
    /// proved counterpart of `run_path_query`.
    pub fn verify_path_query(
        proof: &[u8],
        path_query: &PathQuery,
        grove_version: &GroveVersion,
    ) -> Result<VerifiedPathQuery, Error> {
        let validated = ValidatedPathQuery::for_verification(path_query, grove_version)?;
        let path_query = validated.query();
        match validated.shape() {
            PathQueryShape::KeySelection | PathQueryShape::CountOffsetPaginated { .. } => {
                let (root_hash, elements) = Self::verify_query(proof, path_query, grove_version)?;
                Ok(VerifiedPathQuery::Elements {
                    root_hash,
                    elements,
                })
            }
            PathQueryShape::AggregateLeaf { kind, .. } => match kind {
                AggregateKind::Count => {
                    let (root_hash, count) =
                        Self::verify_aggregate_count_query(proof, path_query, grove_version)?;
                    Ok(VerifiedPathQuery::AggregateCount { root_hash, count })
                }
                AggregateKind::Sum => {
                    let (root_hash, sum) =
                        Self::verify_aggregate_sum_query(proof, path_query, grove_version)?;
                    Ok(VerifiedPathQuery::AggregateSum { root_hash, sum })
                }
                AggregateKind::CountAndSum => {
                    let (root_hash, count, sum) = Self::verify_aggregate_count_and_sum_query(
                        proof,
                        path_query,
                        grove_version,
                    )?;
                    Ok(VerifiedPathQuery::AggregateCountAndSum {
                        root_hash,
                        count,
                        sum,
                    })
                }
            },
            PathQueryShape::AggregateCarrier { kind, .. } => match kind {
                AggregateKind::Count => {
                    let (root_hash, per_key) = Self::verify_aggregate_count_query_per_key(
                        proof,
                        path_query,
                        grove_version,
                    )?;
                    Ok(VerifiedPathQuery::AggregatePerKey {
                        root_hash,
                        per_key: count_per_key_entries(&per_key),
                    })
                }
                AggregateKind::Sum => {
                    let (root_hash, per_key) =
                        Self::verify_aggregate_sum_query_per_key(proof, path_query, grove_version)?;
                    Ok(VerifiedPathQuery::AggregatePerKey {
                        root_hash,
                        per_key: sum_per_key_entries(&per_key),
                    })
                }
                AggregateKind::CountAndSum => {
                    let (root_hash, per_key) = Self::verify_aggregate_count_and_sum_query_per_key(
                        proof,
                        path_query,
                        grove_version,
                    )?;
                    Ok(VerifiedPathQuery::AggregatePerKey {
                        root_hash,
                        per_key: count_and_sum_per_key_entries(&per_key),
                    })
                }
            },
            PathQueryShape::AxisRead { axis } => {
                let (root_hash, trios, outcomes) =
                    Self::verify_axis_shape_walk(proof, &validated, grove_version)?;
                let AxisWalkOutcome { path, result } = match single_outcome(outcomes) {
                    Ok(outcome) => outcome,
                    // No axis layer at all: the walk authenticated that
                    // the queried path does not exist, and the answer
                    // is the traversal's empty one. See
                    // `absent_axis_path_into_verified` for why zero
                    // layers can mean nothing else.
                    Err(0) => {
                        return Self::absent_axis_path_into_verified(
                            root_hash, axis, &trios, path_query,
                        );
                    }
                    Err(count) => {
                        return Err(Error::InvalidProof(
                            path_query.clone(),
                            format!(
                                "a single-path axis read must verify exactly one axis layer, \
                                 got {}",
                                count
                            ),
                        ));
                    }
                };
                if path != path_query.path {
                    return Err(Error::InvalidProof(
                        path_query.clone(),
                        "the verified axis layer is not at the queried path".to_string(),
                    ));
                }
                Self::outcome_into_verified(root_hash, result, &axis.traversal, path_query)
            }
            PathQueryShape::BranchedAxisRead {
                branch_items,
                suffix,
                axis,
            } => {
                let (root_hash, trios, outcomes) =
                    Self::verify_axis_shape_walk(proof, &validated, grove_version)?;

                // A non-entry-listing branched traversal never reaches
                // here: `classify` rejects it as `InvalidQuery` before
                // this function runs, which is where the rule belongs —
                // the trusted reader gets the identical rejection.
                let _ = axis;
                Self::branched_axis_into_verified(
                    root_hash,
                    &trios,
                    outcomes,
                    branch_items,
                    suffix,
                    path_query,
                )
            }
            PathQueryShape::SumBudget { .. } => {
                let proof_v1 =
                    validated.require_v1_envelope(decode_grovedb_proof_canonical(proof)?)?;
                let (root_hash, _, outcomes) =
                    Self::verify_proof_v1_with_axis_outcomes(&proof_v1, path_query, grove_version)?;
                let outcome = match single_outcome(outcomes) {
                    Ok(outcome) => outcome,
                    Err(count) => {
                        return Err(Error::InvalidProof(
                            path_query.clone(),
                            format!(
                                "a sum-budget read must verify exactly one window layer, got {}",
                                count
                            ),
                        ));
                    }
                };
                let AxisWalkOutcome { path, result } = outcome;
                if path != path_query.path {
                    return Err(Error::InvalidProof(
                        path_query.clone(),
                        "the verified sum-budget window is not at the queried path".to_string(),
                    ));
                }
                let AxisWalkResult::SumBudget {
                    matches,
                    total,
                    stop,
                } = result
                else {
                    return Err(Error::InvalidProof(
                        path_query.clone(),
                        "the verified outcome does not match the query's sum-budget shape"
                            .to_string(),
                    ));
                };
                Ok(VerifiedPathQuery::SumBudget {
                    root_hash,
                    matches,
                    total,
                    stop,
                })
            }
        }
    }

    /// Decode + envelope-gate + walk for the axis shapes.
    fn verify_axis_shape_walk(
        proof: &[u8],
        validated: &ValidatedPathQuery<'_>,
        grove_version: &GroveVersion,
    ) -> Result<
        (
            CryptoHash,
            Vec<PathKeyOptionalElementTrio>,
            Vec<AxisWalkOutcome>,
        ),
        Error,
    > {
        let proof_v1 = validated.require_v1_envelope(decode_grovedb_proof_canonical(proof)?)?;
        Self::verify_proof_v1_with_axis_outcomes(&proof_v1, validated.query(), grove_version)
    }

    /// The answer of a single-path axis read whose walk verified with
    /// **no** axis layer: the queried path does not exist, and the
    /// traversal's empty answer is what the proof authenticates.
    ///
    /// Zero layers can mean nothing else. The walk fails closed on every
    /// other way of reaching the end without an axis layer: a present
    /// non-empty tree on the path must carry its lower layer, the
    /// axis-read position itself must be an indexed tree with an axis
    /// layer (a non-indexed element or a missing layer is a hard error,
    /// never a silent absence), and a present-but-empty ancestor is
    /// bound to `NULL_HASH` so a populated tree cannot be rewritten as
    /// empty to hide the descent. What remains is a path segment the
    /// parent's single-key Merk proof authenticated as absent — or an
    /// empty ancestor below which nothing can exist — and that is the
    /// prover's "absent path" too (`GroveDb::path_is_present` decides
    /// it against the proof's own snapshot before the proof is handed
    /// out). The one state the walk reports instead of refusing — a
    /// present non-tree element somewhere on the path, surfaced as a
    /// result row — is rejected here: an axis read has no meaning
    /// under an item, and the prover refuses to generate it.
    ///
    /// Only entry-listing and aggregate traversals have an empty answer.
    /// A rank-of-key read does not (the key has no rank in a tree that
    /// does not exist), so it is rejected exactly as the prover and the
    /// trusted read refuse it.
    fn absent_axis_path_into_verified(
        root_hash: CryptoHash,
        axis: &AxisQuery,
        trios: &[PathKeyOptionalElementTrio],
        path_query: &PathQuery,
    ) -> Result<VerifiedPathQuery, Error> {
        let present = first_present_trio_on_path(trios, &path_query.path);
        if present < trios.len() {
            let (trio_path, trio_key, _) = &trios[present];
            return Err(present_non_tree_on_path(path_query, trio_path, trio_key));
        }
        match &axis.traversal {
            AxisTraversal::RankedPage { .. } => Ok(VerifiedPathQuery::AxisEntries {
                root_hash,
                entries: AxisEntries::empty_for_axis(axis.axis),
                skipped: Some(0),
            }),
            AxisTraversal::Bounded { .. } => Ok(VerifiedPathQuery::AxisEntries {
                root_hash,
                entries: AxisEntries::empty_for_axis(axis.axis),
                skipped: None,
            }),
            AxisTraversal::AggregateOverValueRange { .. } => Ok(VerifiedPathQuery::AxisAggregate {
                root_hash,
                value: 0,
            }),
            AxisTraversal::RankOfKey { .. } => Err(Error::InvalidProof(
                path_query.clone(),
                "a rank-of-key axis read has no answer over an absent path: the key has no \
                 rank in a tree that does not exist, and the prover refuses to generate such \
                 a proof"
                    .to_string(),
            )),
        }
    }

    /// The answer of a branched axis read from its walk: one entry list
    /// per branch key, in query order, or `None` for a branch key whose
    /// absence the branching-level Merk proof authenticates.
    ///
    /// The outcomes are indexed by branch key first
    /// ([`Self::index_branch_outcomes`]), then each branch item is
    /// resolved against that index ([`Self::resolve_branch`]), stopping
    /// at the first error; finally an outcome no branch item claimed is
    /// an error. Index loops whose bodies are helpers, so no loop returns
    /// early.
    fn branched_axis_into_verified(
        root_hash: CryptoHash,
        trios: &[PathKeyOptionalElementTrio],
        outcomes: Vec<AxisWalkOutcome>,
        branch_items: &[QueryItem],
        suffix: &[Vec<u8>],
        path_query: &PathQuery,
    ) -> Result<VerifiedPathQuery, Error> {
        let mut by_branch_key = Self::index_branch_outcomes(outcomes, suffix, path_query)?;

        // Present-element trios anywhere under a branch key that
        // produced no outcome would mean the prover proved the
        // branch present but omitted its axis layer — hiding
        // entries behind a fake absence. Reject.
        let mut branches = Vec::with_capacity(branch_items.len());
        let mut failure = None;
        let mut failed = false;
        let mut i = 0;
        while i < branch_items.len() && !failed {
            match Self::resolve_branch(&branch_items[i], &mut by_branch_key, trios, path_query) {
                Ok(branch) => branches.push(branch),
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
        if !by_branch_key.is_empty() {
            return Err(Error::InvalidProof(
                path_query.clone(),
                "the proof carries axis layers for branch keys the query does not name".to_string(),
            ));
        }
        Ok(VerifiedPathQuery::BranchedAxisEntries {
            root_hash,
            branches,
        })
    }

    /// Index a branched read's outcomes by their branch key (the segment
    /// right after the branching prefix), in walk order, stopping at the
    /// first outcome outside the prefix/branch/suffix structure or
    /// repeating a branch key.
    ///
    /// The outcomes are consumed in order through [`reversed`] and
    /// `Vec::pop` (the extraction has no model of `Vec`'s owning
    /// iterator).
    fn index_branch_outcomes(
        outcomes: Vec<AxisWalkOutcome>,
        suffix: &[Vec<u8>],
        path_query: &PathQuery,
    ) -> Result<BTreeMap<Vec<u8>, AxisWalkResult>, Error> {
        let mut pending = reversed(outcomes);
        let mut by_branch_key = BTreeMap::new();
        let mut failure = None;
        let mut failed = false;
        while !failed && !pending.is_empty() {
            if let Some(outcome) = pending.pop()
                && let Err(e) =
                    Self::index_branch_outcome(&mut by_branch_key, outcome, suffix, path_query)
            {
                failure = Some(e);
                failed = true;
            }
        }
        match failure {
            Some(e) => Err(e),
            None => Ok(by_branch_key),
        }
    }

    /// One outcome of [`Self::index_branch_outcomes`]: check its path is
    /// `prefix ‖ branch_key ‖ suffix` and record it under `branch_key`,
    /// which no earlier outcome may have used.
    fn index_branch_outcome(
        by_branch_key: &mut BTreeMap<Vec<u8>, AxisWalkResult>,
        outcome: AxisWalkOutcome,
        suffix: &[Vec<u8>],
        path_query: &PathQuery,
    ) -> Result<(), Error> {
        let AxisWalkOutcome { path, result } = outcome;
        let prefix_len = path_query.path.len();
        let expected_len = prefix_len + 1 + suffix.len();
        if path.len() != expected_len
            || segments_differ_at(&path, 0, &path_query.path)
            || segments_differ_at(&path, prefix_len + 1, suffix)
        {
            return Err(Error::InvalidProof(
                path_query.clone(),
                "a verified axis layer sits outside the branched query's \
                 prefix/branch/suffix structure"
                    .to_string(),
            ));
        }
        let branch_key = path[prefix_len].clone();
        if by_branch_key.insert(branch_key, result).is_some() {
            return Err(Error::InvalidProof(
                path_query.clone(),
                "duplicate axis layers for one branch key".to_string(),
            ));
        }
        Ok(())
    }

    /// One branch item of a branched read: its entries (taken out of the
    /// index), or `None` when it has no axis layer and no proven-present
    /// trio at or under it.
    fn resolve_branch(
        item: &QueryItem,
        by_branch_key: &mut BTreeMap<Vec<u8>, AxisWalkResult>,
        trios: &[PathKeyOptionalElementTrio],
        path_query: &PathQuery,
    ) -> Result<(Vec<u8>, Option<AxisEntries>), Error> {
        let QueryItem::Key(branch_key) = item else {
            return Err(Error::CorruptedCodeExecution(
                "branched axis read classified with a non-Key branch item",
            ));
        };
        match by_branch_key.remove(branch_key) {
            Some(AxisWalkResult::Entries { entries, .. }) => {
                Ok((branch_key.clone(), Some(entries)))
            }
            Some(_) => Err(Error::InvalidProof(
                path_query.clone(),
                "branched axis reads verify entry-listing traversals only".to_string(),
            )),
            None => {
                // No axis layer for this branch: the
                // branching-level Merk proof must have
                // authenticated the key's ABSENCE. Any
                // present-element trio for the branch key
                // itself — or anywhere under its subtree —
                // means the proof shows the branch present
                // while withholding its axis layer, i.e. a
                // hidden branch dressed up as absence.
                if branch_proven_present(trios, &path_query.path, branch_key) {
                    return Err(Error::InvalidProof(
                        path_query.clone(),
                        format!(
                            "branch key {} is proven present but its axis layer \
                             is missing from the proof",
                            hex::encode(branch_key),
                        ),
                    ));
                }
                Ok((branch_key.clone(), None))
            }
        }
    }

    /// Map a single-path axis outcome into the public result, checking
    /// the outcome family matches the traversal.
    fn outcome_into_verified(
        root_hash: CryptoHash,
        result: AxisWalkResult,
        traversal: &AxisTraversal,
        path_query: &PathQuery,
    ) -> Result<VerifiedPathQuery, Error> {
        match (result, traversal) {
            (
                AxisWalkResult::Entries { entries, skipped },
                AxisTraversal::RankedPage { .. } | AxisTraversal::Bounded { .. },
            ) => Ok(VerifiedPathQuery::AxisEntries {
                root_hash,
                entries,
                skipped,
            }),
            (AxisWalkResult::Rank { rank }, AxisTraversal::RankOfKey { .. }) => {
                Ok(VerifiedPathQuery::AxisRank { root_hash, rank })
            }
            (
                AxisWalkResult::Aggregate { value },
                AxisTraversal::AggregateOverValueRange { .. },
            ) => Ok(VerifiedPathQuery::AxisAggregate { root_hash, value }),
            _ => Err(Error::InvalidProof(
                path_query.clone(),
                "the verified axis outcome does not match the query's traversal".to_string(),
            )),
        }
    }
}

/// The only outcome, or the number of outcomes when there is not exactly
/// one: `<[AxisWalkOutcome; 1]>::try_from(outcomes)` with the error side
/// reduced to the length, the only thing its callers read from it. (The
/// `None` arm cannot happen after the length check.)
fn single_outcome(mut outcomes: Vec<AxisWalkOutcome>) -> Result<AxisWalkOutcome, usize> {
    if outcomes.len() != 1 {
        return Err(outcomes.len());
    }
    match outcomes.pop() {
        Some(outcome) => Ok(outcome),
        None => Err(0),
    }
}

/// `items` in reverse order, so that popping the result yields them in
/// their original order.
fn reversed<T>(mut items: Vec<T>) -> Vec<T> {
    let mut out = Vec::with_capacity(items.len());
    while let Some(item) = items.pop() {
        out.push(item);
    }
    out
}

/// Whether `a[a_start..]` and `b` differ anywhere they overlap:
/// `a[a_start..].iter().zip(b).any(|(x, y)| x != y)` as an index loop.
/// The caller guarantees `a_start <= a.len()`, as the slice did.
fn segments_differ_at(a: &[Vec<u8>], a_start: usize, b: &[Vec<u8>]) -> bool {
    let a_rest = a.len() - a_start;
    let n = if a_rest < b.len() { a_rest } else { b.len() };
    let mut j = 0;
    while j < n && a[a_start + j] == b[j] {
        j += 1;
    }
    j < n
}

/// Whether some present-element trio is the branch key at the branching
/// level (`path` / `branch_key`) or lies anywhere under `path ‖
/// branch_key`: the `trios.iter().any(..)` of the branched read, as an
/// index loop.
fn branch_proven_present(
    trios: &[PathKeyOptionalElementTrio],
    path: &[Vec<u8>],
    branch_key: &[u8],
) -> bool {
    let mut i = 0;
    while i < trios.len() && !trio_shows_branch(&trios[i], path, branch_key) {
        i += 1;
    }
    i < trios.len()
}

/// One trio of [`branch_proven_present`].
fn trio_shows_branch(
    trio: &PathKeyOptionalElementTrio,
    path: &[Vec<u8>],
    branch_key: &[u8],
) -> bool {
    let (trio_path, trio_key, element) = trio;
    if element.is_none() {
        return false;
    }
    // The branch key at the branching level…
    (trio_path.as_slice() == path && trio_key.as_slice() == branch_key)
        // …or anything below prefix/branch_key.
        || (trio_path.len() > path.len()
            && !segments_differ_at(trio_path, 0, path)
            && trio_path[path.len()].as_slice() == branch_key)
}

/// Index of the first present-element trio that sits on the queried
/// `path` (a proper prefix of it, keyed by the next segment), or
/// `trios.len()` if there is none: the `trios.iter().find(..)` of
/// `absent_axis_path_into_verified`, as an index loop.
fn first_present_trio_on_path(trios: &[PathKeyOptionalElementTrio], path: &[Vec<u8>]) -> usize {
    let mut i = 0;
    while i < trios.len() && !trio_on_path(&trios[i], path) {
        i += 1;
    }
    i
}

/// One trio of [`first_present_trio_on_path`].
fn trio_on_path(trio: &PathKeyOptionalElementTrio, path: &[Vec<u8>]) -> bool {
    let (trio_path, trio_key, element) = trio;
    element.is_some()
        && trio_path.len() < path.len()
        && !segments_differ_at(path, 0, trio_path)
        && path[trio_path.len()] == *trio_key
}

/// The error for a single-path axis read whose queried path runs through
/// a present non-tree element. Error construction only (the path is
/// rendered as hex segments), out of line and `#[cold]`; the extraction
/// treats it as opaque.
#[cold]
fn present_non_tree_on_path(
    path_query: &PathQuery,
    trio_path: &[Vec<u8>],
    trio_key: &[u8],
) -> Error {
    Error::InvalidProof(
        path_query.clone(),
        format!(
            "the queried path runs through a present non-tree element at {}/{}; an \
             axis read has no meaning there, so the path is not absent and the \
             proof cannot answer the query",
            trio_path
                .iter()
                .map(hex::encode)
                .collect::<Vec<_>>()
                .join("/"),
            hex::encode(trio_key),
        ),
    )
}

/// `(Vec<u8>, u64)` per-key results as [`VerifiedPathQuery::AggregatePerKey`]
/// entries, in order: the `into_iter().map(..).collect()` as an index loop.
fn count_per_key_entries(per_key: &[(Vec<u8>, u64)]) -> Vec<(Vec<u8>, Option<u64>, Option<i64>)> {
    let mut out = Vec::with_capacity(per_key.len());
    let mut i = 0;
    while i < per_key.len() {
        let entry = &per_key[i];
        out.push((entry.0.clone(), Some(entry.1), None));
        i += 1;
    }
    out
}

/// `(Vec<u8>, i64)` per-key results as [`VerifiedPathQuery::AggregatePerKey`]
/// entries, in order: the `into_iter().map(..).collect()` as an index loop.
fn sum_per_key_entries(per_key: &[(Vec<u8>, i64)]) -> Vec<(Vec<u8>, Option<u64>, Option<i64>)> {
    let mut out = Vec::with_capacity(per_key.len());
    let mut i = 0;
    while i < per_key.len() {
        let entry = &per_key[i];
        out.push((entry.0.clone(), None, Some(entry.1)));
        i += 1;
    }
    out
}

/// `(Vec<u8>, u64, i64)` per-key results as [`VerifiedPathQuery::AggregatePerKey`]
/// entries, in order: the `into_iter().map(..).collect()` as an index loop.
fn count_and_sum_per_key_entries(
    per_key: &[(Vec<u8>, u64, i64)],
) -> Vec<(Vec<u8>, Option<u64>, Option<i64>)> {
    let mut out = Vec::with_capacity(per_key.len());
    let mut i = 0;
    while i < per_key.len() {
        let entry = &per_key[i];
        out.push((entry.0.clone(), Some(entry.1), Some(entry.2)));
        i += 1;
    }
    out
}

/// Differential tests for the pieces of `verify_path_query` reshaped for
/// the Rust-to-Lean extraction. Each `ref_` function is the previous code,
/// verbatim except for its name and, for the branched read, being lifted
/// out of its `match` arm into a function of the arm's inputs.
#[cfg(test)]
mod reference_tests {
    use grovedb_merk::proofs::query::IndexAxis;

    use super::*;
    use crate::Element;

    #[allow(clippy::too_many_arguments)]
    fn ref_branched(
        root_hash: CryptoHash,
        trios: Vec<PathKeyOptionalElementTrio>,
        outcomes: Vec<AxisWalkOutcome>,
        branch_items: &[QueryItem],
        suffix: &[Vec<u8>],
        path_query: &PathQuery,
    ) -> Result<VerifiedPathQuery, Error> {
        // Index outcomes by their branch key (the segment right
        // after the branching prefix).
        let prefix_len = path_query.path.len();
        let mut by_branch_key: std::collections::BTreeMap<Vec<u8>, AxisWalkResult> =
            std::collections::BTreeMap::new();
        for outcome in outcomes {
            let expected_len = prefix_len + 1 + suffix.len();
            if outcome.path.len() != expected_len
                || outcome.path[..prefix_len]
                    .iter()
                    .zip(&path_query.path)
                    .any(|(a, b)| a != b)
                || outcome.path[prefix_len + 1..]
                    .iter()
                    .zip(suffix)
                    .any(|(a, b)| a != b)
            {
                return Err(Error::InvalidProof(
                    path_query.clone(),
                    "a verified axis layer sits outside the branched query's \
                     prefix/branch/suffix structure"
                        .to_string(),
                ));
            }
            let branch_key = outcome.path[prefix_len].clone();
            if by_branch_key.insert(branch_key, outcome.result).is_some() {
                return Err(Error::InvalidProof(
                    path_query.clone(),
                    "duplicate axis layers for one branch key".to_string(),
                ));
            }
        }

        // Present-element trios anywhere under a branch key that
        // produced no outcome would mean the prover proved the
        // branch present but omitted its axis layer — hiding
        // entries behind a fake absence. Reject.
        let mut branches = Vec::with_capacity(branch_items.len());
        for item in branch_items {
            let grovedb_merk::proofs::query::query_item::QueryItem::Key(branch_key) = item else {
                return Err(Error::CorruptedCodeExecution(
                    "branched axis read classified with a non-Key branch item",
                ));
            };
            match by_branch_key.remove(branch_key) {
                Some(AxisWalkResult::Entries { entries, .. }) => {
                    branches.push((branch_key.clone(), Some(entries)));
                }
                Some(_) => {
                    return Err(Error::InvalidProof(
                        path_query.clone(),
                        "branched axis reads verify entry-listing traversals only".to_string(),
                    ));
                }
                None => {
                    // No axis layer for this branch: the
                    // branching-level Merk proof must have
                    // authenticated the key's ABSENCE. Any
                    // present-element trio for the branch key
                    // itself — or anywhere under its subtree —
                    // means the proof shows the branch present
                    // while withholding its axis layer, i.e. a
                    // hidden branch dressed up as absence.
                    let present = trios.iter().any(|(trio_path, trio_key, element)| {
                        if element.is_none() {
                            return false;
                        }
                        // The branch key at the branching level…
                        (trio_path == &path_query.path && trio_key == branch_key)
                            // …or anything below prefix/branch_key.
                            || (trio_path.len() > path_query.path.len()
                                && trio_path[..path_query.path.len()] == path_query.path
                                && trio_path[path_query.path.len()] == *branch_key)
                    });
                    if present {
                        return Err(Error::InvalidProof(
                            path_query.clone(),
                            format!(
                                "branch key {} is proven present but its axis layer \
                                 is missing from the proof",
                                hex::encode(branch_key),
                            ),
                        ));
                    }
                    branches.push((branch_key.clone(), None));
                }
            }
        }
        if !by_branch_key.is_empty() {
            return Err(Error::InvalidProof(
                path_query.clone(),
                "the proof carries axis layers for branch keys the query does not name".to_string(),
            ));
        }
        Ok(VerifiedPathQuery::BranchedAxisEntries {
            root_hash,
            branches,
        })
    }

    fn ref_absent_axis_path_into_verified(
        root_hash: CryptoHash,
        axis: &AxisQuery,
        trios: &[PathKeyOptionalElementTrio],
        path_query: &PathQuery,
    ) -> Result<VerifiedPathQuery, Error> {
        let on_the_queried_path = |trio_path: &[Vec<u8>], trio_key: &[u8]| {
            trio_path.len() < path_query.path.len()
                && path_query.path[..trio_path.len()] == *trio_path
                && path_query.path[trio_path.len()] == *trio_key
        };
        if let Some((trio_path, trio_key, _)) =
            trios.iter().find(|(trio_path, trio_key, element)| {
                element.is_some() && on_the_queried_path(trio_path, trio_key)
            })
        {
            return Err(Error::InvalidProof(
                path_query.clone(),
                format!(
                    "the queried path runs through a present non-tree element at {}/{}; an \
                     axis read has no meaning there, so the path is not absent and the \
                     proof cannot answer the query",
                    trio_path
                        .iter()
                        .map(hex::encode)
                        .collect::<Vec<_>>()
                        .join("/"),
                    hex::encode(trio_key),
                ),
            ));
        }
        match &axis.traversal {
            AxisTraversal::RankedPage { .. } => Ok(VerifiedPathQuery::AxisEntries {
                root_hash,
                entries: AxisEntries::empty_for_axis(axis.axis),
                skipped: Some(0),
            }),
            AxisTraversal::Bounded { .. } => Ok(VerifiedPathQuery::AxisEntries {
                root_hash,
                entries: AxisEntries::empty_for_axis(axis.axis),
                skipped: None,
            }),
            AxisTraversal::AggregateOverValueRange { .. } => Ok(VerifiedPathQuery::AxisAggregate {
                root_hash,
                value: 0,
            }),
            AxisTraversal::RankOfKey { .. } => Err(Error::InvalidProof(
                path_query.clone(),
                "a rank-of-key axis read has no answer over an absent path: the key has no \
                 rank in a tree that does not exist, and the prover refuses to generate such \
                 a proof"
                    .to_string(),
            )),
        }
    }

    fn seg(s: &[u8]) -> Vec<u8> {
        s.to_vec()
    }

    /// Outcome `i` of a fixed pool for a branched read over `prefix` and
    /// `suffix`: well-formed paths `prefix/k/suffix` for three branch keys
    /// and each result family, and paths off the structure (too short, a
    /// wrong prefix segment, a wrong suffix segment).
    fn outcome(i: usize, prefix: &[Vec<u8>], suffix: &[Vec<u8>]) -> AxisWalkOutcome {
        let entries = |axis| AxisWalkResult::Entries {
            entries: AxisEntries::empty_for_axis(axis),
            skipped: None,
        };
        let at = |k: &[u8]| [prefix, &[seg(k)][..], suffix].concat();
        let (path, result) = match i {
            0 => (at(b"a"), entries(IndexAxis::Count)),
            1 => (at(b"b"), entries(IndexAxis::Sum)),
            2 => (at(b"c"), entries(IndexAxis::Avg)),
            3 => (at(b"a"), AxisWalkResult::Rank { rank: 1 }),
            4 => (
                at(b"b")[..prefix.len() + 1].to_vec(),
                entries(IndexAxis::Count),
            ),
            5 => {
                // A wrong segment in the prefix, or (for an empty prefix)
                // one segment too many in front.
                let mut path = at(b"a");
                if prefix.is_empty() {
                    path.insert(0, seg(b"x"));
                } else {
                    path[0] = seg(b"x");
                }
                (path, entries(IndexAxis::Count))
            }
            _ => {
                let mut path = at(b"a");
                let last = path.len() - 1;
                path[last] = seg(b"t");
                (path, entries(IndexAxis::Count))
            }
        };
        AxisWalkOutcome { path, result }
    }

    const POOL: usize = 7;

    fn outcome_lists() -> Vec<Vec<usize>> {
        let mut lists = vec![vec![]];
        let mut last = vec![vec![]];
        for _ in 0..3 {
            let mut longer = Vec::new();
            for l in &last {
                for i in 0..POOL {
                    let mut l2: Vec<usize> = l.clone();
                    l2.push(i);
                    longer.push(l2);
                }
            }
            lists.extend(longer.iter().cloned());
            last = longer;
        }
        lists
    }

    fn branch_item_lists() -> Vec<Vec<QueryItem>> {
        let key = |k: &[u8]| QueryItem::Key(k.to_vec());
        let range = || QueryItem::Range(b"a".to_vec()..b"b".to_vec());
        vec![
            vec![],
            vec![key(b"a")],
            vec![key(b"a"), key(b"b")],
            vec![key(b"b"), key(b"a"), key(b"c")],
            vec![key(b"a"), key(b"a")],
            vec![key(b"d")],
            vec![range()],
            vec![key(b"d"), range()],
            vec![range(), key(b"d")],
            vec![key(b"a"), range(), key(b"b")],
        ]
    }

    /// Trio lists over `prefix`: absent and present rows for a branch key
    /// at the branching level, present rows under a branch, and rows off
    /// the prefix.
    fn trio_lists(prefix: &[Vec<u8>]) -> Vec<Vec<PathKeyOptionalElementTrio>> {
        let item = || Some(Element::new_item(b"v".to_vec()));
        let p = || prefix.to_vec();
        let under = |k: &[u8], rest: &[&[u8]]| {
            let mut path = p();
            path.push(seg(k));
            for r in rest {
                path.push(seg(r));
            }
            path
        };
        let mut off = p();
        off.insert(0, seg(b"q"));
        vec![
            vec![],
            vec![(p(), seg(b"a"), None)],
            vec![(p(), seg(b"a"), item())],
            vec![(p(), seg(b"d"), item())],
            vec![(under(b"d", &[b"s"]), seg(b"x"), item())],
            vec![(under(b"d", &[]), seg(b"x"), None)],
            vec![(off, seg(b"d"), item()), (p(), seg(b"b"), item())],
            vec![(vec![], seg(b"r"), item())],
        ]
    }

    #[test]
    fn branched_read_matches_reference() {
        let shapes = [
            (vec![seg(b"r")], vec![seg(b"s")]),
            (vec![], vec![seg(b"s")]),
            (vec![seg(b"r"), seg(b"q")], vec![seg(b"s"), seg(b"u")]),
        ];
        let mut cases = 0;
        for (prefix, suffix) in &shapes {
            let path_query = PathQuery::new_single_key(prefix.clone(), b"k".to_vec());
            for outcomes in outcome_lists() {
                for items in branch_item_lists() {
                    for trios in trio_lists(prefix) {
                        let got = GroveDb::branched_axis_into_verified(
                            [1; 32],
                            &trios,
                            outcomes
                                .iter()
                                .map(|&i| outcome(i, prefix, suffix))
                                .collect(),
                            &items,
                            suffix,
                            &path_query,
                        );
                        let want = ref_branched(
                            [1; 32],
                            trios.clone(),
                            outcomes
                                .iter()
                                .map(|&i| outcome(i, prefix, suffix))
                                .collect(),
                            &items,
                            suffix,
                            &path_query,
                        );
                        assert_eq!(
                            format!("{got:?}"),
                            format!("{want:?}"),
                            "{outcomes:?} {items:?} {trios:?}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases > 90_000, "{cases}");
    }

    #[test]
    fn absent_axis_path_matches_reference() {
        let traversals = [
            AxisTraversal::RankedPage { k: 1, offset: 0 },
            AxisTraversal::Bounded {
                lo: 0,
                hi: 1,
                limit: 1,
            },
            AxisTraversal::RankOfKey { key: b"k".to_vec() },
            AxisTraversal::AggregateOverValueRange {
                lo: 0,
                hi: 1,
                fold: grovedb_merk::proofs::query::AggregateFold::Total,
            },
        ];
        let item = || Some(Element::new_item(b"v".to_vec()));
        let trio_lists: Vec<Vec<PathKeyOptionalElementTrio>> = vec![
            vec![],
            vec![(vec![], seg(b"a"), item())],
            vec![(vec![], seg(b"a"), None)],
            vec![
                (vec![], seg(b"z"), item()),
                (vec![seg(b"a")], seg(b"b"), item()),
            ],
            vec![
                (vec![seg(b"a")], seg(b"z"), item()),
                (vec![seg(b"a"), seg(b"b")], seg(b"c"), item()),
            ],
            vec![(vec![seg(b"a"), seg(b"b")], seg(b"c"), item())],
            vec![(vec![seg(b"x")], seg(b"b"), item())],
        ];
        let paths = [
            vec![],
            vec![seg(b"a")],
            vec![seg(b"a"), seg(b"b")],
            vec![seg(b"a"), seg(b"b"), seg(b"c")],
        ];
        for path in paths {
            let path_query = PathQuery::new_single_key(path, b"k".to_vec());
            for traversal in &traversals {
                let axis = AxisQuery {
                    axis: IndexAxis::Count,
                    traversal: traversal.clone(),
                    descending: false,
                    projection: grovedb_merk::proofs::query::AxisProjection::Entries,
                };
                for trios in &trio_lists {
                    let got =
                        GroveDb::absent_axis_path_into_verified([2; 32], &axis, trios, &path_query);
                    let want =
                        ref_absent_axis_path_into_verified([2; 32], &axis, trios, &path_query);
                    assert_eq!(format!("{got:?}"), format!("{want:?}"), "{trios:?}");
                }
            }
        }
    }

    #[test]
    fn single_outcome_matches_array_try_from() {
        for n in 0..4 {
            let one = |i| outcome(i, &[], &[]);
            let got = single_outcome((0..n).map(one).collect());
            let want = <[AxisWalkOutcome; 1]>::try_from((0..n).map(one).collect::<Vec<_>>());
            match (got, want) {
                (Ok(a), Ok([b])) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                (Err(len), Err(v)) => assert_eq!(len, v.len()),
                (a, b) => panic!("{a:?} vs {b:?}"),
            }
        }
    }

    #[test]
    fn per_key_entries_match_iterator_maps() {
        let count: Vec<(Vec<u8>, u64)> = vec![(seg(b"a"), 1), (seg(b"b"), 2)];
        let want: Vec<_> = count
            .clone()
            .into_iter()
            .map(|(key, count)| (key, Some(count), None))
            .collect();
        assert_eq!(count_per_key_entries(&count), want);
        let sum: Vec<(Vec<u8>, i64)> = vec![(seg(b"a"), -1), (seg(b"b"), 2)];
        let want: Vec<_> = sum
            .clone()
            .into_iter()
            .map(|(key, sum)| (key, None, Some(sum)))
            .collect();
        assert_eq!(sum_per_key_entries(&sum), want);
        let both: Vec<(Vec<u8>, u64, i64)> = vec![(seg(b"a"), 1, -1), (seg(b"b"), 2, 2)];
        let want: Vec<_> = both
            .clone()
            .into_iter()
            .map(|(key, count, sum)| (key, Some(count), Some(sum)))
            .collect();
        assert_eq!(count_and_sum_per_key_entries(&both), want);
        assert!(count_per_key_entries(&[]).is_empty());
    }

    #[test]
    fn segments_differ_at_matches_zip_any() {
        let lists: Vec<Vec<Vec<u8>>> = vec![
            vec![],
            vec![seg(b"a")],
            vec![seg(b"b")],
            vec![seg(b"a"), seg(b"b")],
            vec![seg(b"a"), seg(b"c")],
            vec![seg(b"b"), seg(b"a"), seg(b"c")],
        ];
        for a in &lists {
            for b in &lists {
                for start in 0..=a.len() {
                    assert_eq!(
                        segments_differ_at(a, start, b),
                        a[start..].iter().zip(b).any(|(x, y)| x != y),
                        "{a:?} {start} {b:?}"
                    );
                }
            }
        }
    }
}
