//! Borrowed validation shared by the read, prove and verify dispatchers.
//!
//! Shape grammar lives in `PathQuery::classify`; this module adds only the
//! consumer's version/envelope eligibility. It performs no storage access,
//! serialization, query rewriting, or cost accounting. Tree types and proof
//! contents must still be checked by the execution/verification walkers.
//!
//! The legacy result-specific APIs retain their narrower contracts: scalar
//! aggregate APIs require leaves, element APIs cannot return read-mode
//! results, ordinary reads permit offsets that proofs cannot attest, and
//! absence/parent-info APIs have their own limit/offset requirements. Those
//! restrictions are not additional query shapes.

#[cfg(feature = "minimal")]
use grovedb_version::error::GroveVersionError;
use grovedb_version::version::{FeatureVersion, GroveVersion};

#[cfg(feature = "minimal")]
use super::AggregateKind;
use super::PathQueryShape;
use crate::{
    operations::proof::{GroveDBProof, GroveDBProofV1},
    Error, PathQuery,
};

/// A query and its validated, allocation-free shape at a particular version.
/// Aggregate kind and inner range are retained in `shape`, not rediscovered
/// from the query by consumers. This type is deliberately not serialized.
///
/// `query` is declared last: in the extracted Lean structure a field named
/// `query` would shadow the `query` namespace in the types of later fields.
pub(crate) struct ValidatedPathQuery<'q> {
    shape: PathQueryShape<'q>,
    grove_version: &'q GroveVersion,
    query: &'q PathQuery,
}

impl<'q> ValidatedPathQuery<'q> {
    pub(crate) fn new(
        path_query: &'q PathQuery,
        grove_version: &'q GroveVersion,
    ) -> Result<Self, Error> {
        Ok(Self {
            query: path_query,
            shape: path_query.classify()?,
            grove_version,
        })
    }

    #[cfg(feature = "minimal")]
    pub(crate) fn for_read(
        query: &'q PathQuery,
        grove_version: &'q GroveVersion,
    ) -> Result<Self, Error> {
        let validated = Self::new(query, grove_version)?;
        // Exhaustive over the shapes so adding one forces a deliberate
        // read-side version-policy decision here.
        match validated.shape {
            PathQueryShape::AxisRead { .. }
            | PathQueryShape::BranchedAxisRead { .. }
            | PathQueryShape::SumBudget { .. } => {
                match grove_version.grovedb_versions.path_query_methods.unified_read_mode {
                    0 => return Err(Error::NotSupported(
                        "read-mode (axis / sum-budget) path queries are not served at this grove version"
                            .to_string(),
                    )),
                    1 => {}
                    received => return Err(Error::VersionError(
                        GroveVersionError::UnknownVersionMismatch {
                            method: "run_path_query (unified_read_mode)".to_string(),
                            known_versions: vec![0, 1],
                            received,
                        },
                    )),
                }
            }
            // Served at every version, exactly as their dedicated entry
            // points serve them.
            PathQueryShape::KeySelection
            | PathQueryShape::CountOffsetPaginated { .. }
            | PathQueryShape::AggregateLeaf { .. }
            | PathQueryShape::AggregateCarrier { .. } => {}
        }
        Ok(validated)
    }

    #[cfg(feature = "minimal")]
    pub(crate) fn for_proof(
        query: &'q PathQuery,
        grove_version: &'q GroveVersion,
    ) -> Result<Self, Error> {
        // Preserve the current prover's per-instance gate before shape
        // validation, including its historical error precedence.
        query.reject_unserved_per_instance_limits(grove_version)?;
        let prove_version = grove_version
            .grovedb_versions
            .operations
            .proof
            .prove_query_non_serialized;
        let shape = query.classify_with_offset_gate(|| match prove_version {
            // V0 rejected offsets before inspecting pagination syntax.
            0 => {
                query.reject_per_instance_limits("the V0 prover")?;
                Err(Error::InvalidQuery(
                    "proved path queries can not have offsets",
                ))
            }
            1 => Ok(()),
            received => Err(unknown_prove_version(received)),
        })?;
        let validated = Self {
            query,
            shape,
            grove_version,
        };
        if prove_version == 0 {
            // Exhaustive over the shapes so adding one forces a
            // deliberate V0-envelope decision here (fail closed).
            let subject = match shape {
                PathQueryShape::AxisRead { .. } | PathQueryShape::BranchedAxisRead { .. } => {
                    Some("axis-ordered path queries".to_string())
                }
                PathQueryShape::SumBudget { .. } => Some("sum-budget path queries".to_string()),
                PathQueryShape::AggregateLeaf { kind, .. }
                | PathQueryShape::AggregateCarrier { kind, .. } => {
                    Some(format!("{} proofs", kind.proof_family_name()))
                }
                // Key selection is what the V0 envelope serves; the
                // count-offset shape cannot reach here (the offset gate
                // above already rejected it for the V0 prover).
                PathQueryShape::KeySelection | PathQueryShape::CountOffsetPaginated { .. } => None,
            };
            if let Some(subject) = subject {
                return Err(Error::NotSupported(format!(
                    "{subject} require V1 proof envelopes; upgrade the grove version producing \
                     the proof"
                )));
            }
        }
        validated.check_proof_capabilities(true)?;
        match prove_version {
            // The V0 prover's frozen limit accounting predates
            // per-instance caps. Redundant for every shipped version
            // table (the unserved gate above already fired), kept for
            // the historical precedence of a hypothetical table that
            // serves the caps while selecting the V0 prover.
            0 => query.reject_per_instance_limits("the V0 prover")?,
            1 => {}
            received => return Err(unknown_prove_version(received)),
        }
        // limit == 0 is checked by the prover AFTER the count-offset
        // target-tree open. Moving it here would change operation costs.
        Ok(validated)
    }

    pub(crate) fn for_verification(
        path_query: &'q PathQuery,
        grove_version: &'q GroveVersion,
    ) -> Result<Self, Error> {
        let validated = Self::new(path_query, grove_version)?;
        validated.check_proof_capabilities(false)?;
        Ok(validated)
    }

    fn check_proof_capabilities(&self, generating: bool) -> Result<(), Error> {
        let versions = &self.grove_version.grovedb_versions.operations.proof;
        // Exhaustive over the shapes (in `capability_slot`) so adding one
        // forces a deliberate capability-gate decision here (fail closed).
        let message = match capability_slot(&self.shape) {
            CapabilitySlot::AxisDescent => unsupported_message(
                versions.axis_descent_in_v1_envelope,
                generating,
                "axis-ordered descents in the V1 proof envelope are not emitted at this grove version",
                "axis-ordered descents in the V1 proof envelope are not accepted at this grove version",
            ),
            CapabilitySlot::SumBudget => unsupported_message(
                versions.sum_budget_in_v1_envelope,
                generating,
                "sum-budget windows in the V1 proof envelope are not emitted at this grove version",
                "sum-budget windows in the V1 proof envelope are not accepted at this grove version",
            ),
            CapabilitySlot::Unneeded => None,
        };
        match message {
            Some(message) => Err(Error::NotSupported(message.to_string())),
            None => Ok(()),
        }
    }

    /// Decode-time envelope gate for the shapes verified through the
    /// unified axis/sum-budget walkers. Returns the V1 payload so
    /// "accepted ⇒ V1 envelope" is carried by the type rather than a
    /// call-site `unreachable!`. The other shapes verify through their
    /// dedicated walkers, which own their envelope errors — routing
    /// them here is a dispatch bug and fails closed.
    pub(crate) fn require_v1_envelope(
        &self,
        decoded: GroveDBProof,
    ) -> Result<GroveDBProofV1, Error> {
        let message = match self.shape {
            PathQueryShape::AxisRead { .. } | PathQueryShape::BranchedAxisRead { .. } => {
                "axis-ordered path queries require V1 proof envelopes"
            }
            PathQueryShape::SumBudget { .. } => {
                "sum-budget path queries require V1 proof envelopes"
            }
            PathQueryShape::KeySelection
            | PathQueryShape::CountOffsetPaginated { .. }
            | PathQueryShape::AggregateLeaf { .. }
            | PathQueryShape::AggregateCarrier { .. } => {
                return Err(Error::CorruptedCodeExecution(
                    "require_v1_envelope called for a shape verified by its dedicated walker",
                ));
            }
        };
        match decoded {
            GroveDBProof::V1(proof) => Ok(proof),
            GroveDBProof::V0(_) => Err(Error::NotSupported(message.to_string())),
        }
    }

    pub(crate) fn query(&self) -> &'q PathQuery {
        self.query
    }

    pub(crate) fn shape(&self) -> PathQueryShape<'q> {
        self.shape
    }

    /// The aggregate terminal's depth is fixed by the leaf/carrier grammar.
    /// The prover still opens and checks the actual tree at this depth.
    #[cfg(feature = "minimal")]
    pub(crate) fn aggregate_at_depth(
        &self,
        depth: usize,
    ) -> Option<(AggregateKind, &'q grovedb_merk::proofs::query::QueryItem)> {
        match self.shape {
            PathQueryShape::AggregateLeaf { kind, inner } if depth == self.query.path.len() => {
                Some((kind, inner))
            }
            PathQueryShape::AggregateCarrier { kind, inner }
                if depth
                    == self.query.path.len()
                        + 1
                        + self
                            .query
                            .query
                            .query
                            .default_subquery_branch
                            .subquery_path
                            .as_ref()
                            .map_or(0, Vec::len) =>
            {
                Some((kind, inner))
            }
            _ => None,
        }
    }

    #[cfg(feature = "minimal")]
    pub(crate) fn proof_version(&self) -> u16 {
        self.grove_version
            .grovedb_versions
            .operations
            .proof
            .prove_query_non_serialized
    }
}

#[cfg(feature = "minimal")]
fn unknown_prove_version(received: u16) -> Error {
    Error::VersionError(GroveVersionError::UnknownVersionMismatch {
        method: "prove_query_non_serialized".to_string(),
        known_versions: vec![0, 1],
        received,
    })
}

/// The proof capability slot a shape needs beyond the envelope version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapabilitySlot {
    /// `proof.axis_descent_in_v1_envelope`.
    AxisDescent,
    /// `proof.sum_budget_in_v1_envelope`.
    SumBudget,
    /// Serving the shape needs no capability slot beyond the envelope
    /// version itself.
    Unneeded,
}

/// The capability slot `shape` needs. Exhaustive over the shapes so adding
/// one forces a deliberate capability-gate decision (fail closed). A
/// function of its own, so the extracted gate does not join inside the
/// match on the borrowed shape.
fn capability_slot(shape: &PathQueryShape<'_>) -> CapabilitySlot {
    match shape {
        PathQueryShape::AxisRead { .. } | PathQueryShape::BranchedAxisRead { .. } => {
            CapabilitySlot::AxisDescent
        }
        PathQueryShape::SumBudget { .. } => CapabilitySlot::SumBudget,
        PathQueryShape::KeySelection
        | PathQueryShape::CountOffsetPaginated { .. }
        | PathQueryShape::AggregateLeaf { .. }
        | PathQueryShape::AggregateCarrier { .. } => CapabilitySlot::Unneeded,
    }
}

/// The refusal message for a capability slot at `version`: `None` when the
/// slot is served (`version == 1`), else the message for generating or for
/// verifying. Plain `if`/`else`: the extraction does not translate
/// `bool::then_some`.
fn unsupported_message(
    version: FeatureVersion,
    generating: bool,
    emitted: &'static str,
    accepted: &'static str,
) -> Option<&'static str> {
    if version != 1 {
        if generating {
            Some(emitted)
        } else {
            Some(accepted)
        }
    } else {
        None
    }
}

/// `check_proof_capabilities` now chooses its message through
/// `capability_slot` and `unsupported_message`. `ref_message` is the
/// previous `match`, verbatim except that it takes the shape and the
/// version as parameters. Every shape is checked under every combination
/// of the two capability slots, generating and verifying.
#[cfg(test)]
mod capability_reference_tests {
    use grovedb_merk::proofs::query::{
        query_item::QueryItem, AxisProjection, AxisQuery, AxisTraversal, IndexAxis, SumBudgetRead,
    };

    use super::*;
    use crate::query::shape::AggregateKind;

    fn ref_message(
        shape: PathQueryShape<'_>,
        grove_version: &GroveVersion,
        generating: bool,
    ) -> Option<&'static str> {
        let versions = &grove_version.grovedb_versions.operations.proof;
        match shape {
            PathQueryShape::AxisRead { .. } | PathQueryShape::BranchedAxisRead { .. } => {
                (versions.axis_descent_in_v1_envelope != 1).then_some(if generating {
                    "axis-ordered descents in the V1 proof envelope are not emitted at this grove version"
                } else {
                    "axis-ordered descents in the V1 proof envelope are not accepted at this grove version"
                })
            }
            PathQueryShape::SumBudget { .. } => {
                (versions.sum_budget_in_v1_envelope != 1).then_some(if generating {
                    "sum-budget windows in the V1 proof envelope are not emitted at this grove version"
                } else {
                    "sum-budget windows in the V1 proof envelope are not accepted at this grove version"
                })
            }
            PathQueryShape::KeySelection
            | PathQueryShape::CountOffsetPaginated { .. }
            | PathQueryShape::AggregateLeaf { .. }
            | PathQueryShape::AggregateCarrier { .. } => None,
        }
    }

    #[test]
    fn capability_messages_match_reference() {
        let item = QueryItem::Key(b"k".to_vec());
        let items = [item.clone()];
        let suffix = [b"s".to_vec()];
        let axis = AxisQuery {
            axis: IndexAxis::Count,
            traversal: AxisTraversal::RankedPage { k: 1, offset: 0 },
            descending: false,
            projection: AxisProjection::Entries,
        };
        let budget = SumBudgetRead {
            sum_limit: 1,
            match_limit: None,
        };
        let query = PathQuery::new_single_key(vec![], b"k".to_vec());
        let shapes = [
            PathQueryShape::KeySelection,
            PathQueryShape::CountOffsetPaginated { inner: &item },
            PathQueryShape::AggregateLeaf {
                kind: AggregateKind::Count,
                inner: &item,
            },
            PathQueryShape::AggregateCarrier {
                kind: AggregateKind::Sum,
                inner: &item,
            },
            PathQueryShape::AxisRead { axis: &axis },
            PathQueryShape::BranchedAxisRead {
                branch_items: &items,
                suffix: &suffix,
                axis: &axis,
            },
            PathQueryShape::SumBudget {
                budget: &budget,
                items: &items,
            },
        ];
        for shape in shapes {
            for axis_slot in 0..3 {
                for sum_slot in 0..3 {
                    let mut grove_version = GroveVersion::latest().clone();
                    let proof = &mut grove_version.grovedb_versions.operations.proof;
                    proof.axis_descent_in_v1_envelope = axis_slot;
                    proof.sum_budget_in_v1_envelope = sum_slot;
                    let validated = ValidatedPathQuery {
                        query: &query,
                        shape,
                        grove_version: &grove_version,
                    };
                    for generating in [false, true] {
                        let got = match validated.check_proof_capabilities(generating) {
                            Ok(()) => None,
                            Err(e) => Some(format!("{e:?}")),
                        };
                        let want = ref_message(shape, &grove_version, generating)
                            .map(|m| format!("{:?}", Error::NotSupported(m.to_string())));
                        assert_eq!(got, want, "{shape:?} {axis_slot} {sum_slot} {generating}");
                    }
                }
            }
        }
    }
}
