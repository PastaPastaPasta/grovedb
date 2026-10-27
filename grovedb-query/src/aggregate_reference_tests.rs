//! Differential tests for the aggregate validators reshaped for the
//! Rust-to-Lean extraction: each `ref_` function below is the previous body,
//! verbatim except for its name, `q` for `self`, and calling the `ref_` leaf
//! validator. Every generated query must get the same answer (the same
//! borrowed item, or an error with the same variant and message) from both.

use indexmap::IndexMap;

use crate::{error::Error, query::any_empty_key, Query, QueryItem, SubqueryBranch};

// ---- previous bodies ----

fn ref_validate_leaf_aggregate_count_on_range(q: &Query) -> Result<&QueryItem, Error> {
    if q.items.len() != 1 {
        return Err(Error::InvalidOperation(
            "AggregateCountOnRange must be the only item in the query",
        ));
    }
    let inner = match &q.items[0] {
        QueryItem::AggregateCountOnRange(inner) => inner.as_ref(),
        _ => {
            return Err(Error::InvalidOperation(
                "validate_aggregate_count_on_range called on a query without an \
                 AggregateCountOnRange item",
            ));
        }
    };
    match inner {
        QueryItem::Key(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountOnRange may not wrap Key — use has_raw / get_raw for \
                 existence tests",
            ));
        }
        QueryItem::RangeFull(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountOnRange may not wrap RangeFull — read the parent \
                 ProvableCountTree element for the unconditional total",
            ));
        }
        QueryItem::AggregateCountOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountOnRange may not wrap another AggregateCountOnRange",
            ));
        }
        QueryItem::AggregateSumOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountOnRange may not wrap AggregateSumOnRange — the \
                 aggregate variants are orthogonal",
            ));
        }
        QueryItem::AggregateCountAndSumOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountOnRange may not wrap AggregateCountAndSumOnRange — the \
                 aggregate variants are orthogonal",
            ));
        }
        _ => {}
    }
    if q.default_subquery_branch.subquery.is_some()
        || q.default_subquery_branch.subquery_path.is_some()
    {
        return Err(Error::InvalidOperation(
            "AggregateCountOnRange queries may not carry a default subquery branch",
        ));
    }
    if let Some(branches) = &q.conditional_subquery_branches
        && !branches.is_empty()
    {
        return Err(Error::InvalidOperation(
            "AggregateCountOnRange queries may not carry conditional subquery branches",
        ));
    }
    Ok(inner)
}

fn ref_validate_carrier_aggregate_count_on_range(q: &Query) -> Result<&QueryItem, Error> {
    if q.items.is_empty() {
        return Err(Error::InvalidOperation(
            "carrier AggregateCountOnRange query must have at least one outer item",
        ));
    }
    for item in &q.items {
        match item {
            QueryItem::Key(_)
            | QueryItem::Range(_)
            | QueryItem::RangeInclusive(_)
            | QueryItem::RangeFrom(_)
            | QueryItem::RangeTo(_)
            | QueryItem::RangeToInclusive(_)
            | QueryItem::RangeAfter(_)
            | QueryItem::RangeAfterTo(_)
            | QueryItem::RangeAfterToInclusive(_) => {}
            QueryItem::RangeFull(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateCountOnRange query may not have a RangeFull outer item",
                ));
            }
            QueryItem::AggregateCountOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateCountOnRange query may not own an \
                     AggregateCountOnRange item — use the leaf shape instead",
                ));
            }
            QueryItem::AggregateSumOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateCountOnRange query may not own an \
                     AggregateSumOnRange item — the two aggregate variants are orthogonal",
                ));
            }
            QueryItem::AggregateCountAndSumOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateCountOnRange query may not own an \
                     AggregateCountAndSumOnRange item — the aggregate variants are \
                     orthogonal",
                ));
            }
        }
    }
    let subquery = match q.default_subquery_branch.subquery.as_deref() {
        Some(sub) => sub,
        None => {
            return Err(Error::InvalidOperation(
                "carrier AggregateCountOnRange query must set \
                 default_subquery_branch.subquery to a leaf `AggregateCountOnRange` query",
            ));
        }
    };
    if let Some(path) = &q.default_subquery_branch.subquery_path
        && path.iter().any(|k| k.is_empty())
    {
        return Err(Error::InvalidOperation(
            "carrier AggregateCountOnRange query's subquery_path must contain non-empty keys",
        ));
    }
    if let Some(branches) = &q.conditional_subquery_branches
        && !branches.is_empty()
    {
        return Err(Error::InvalidOperation(
            "carrier AggregateCountOnRange query may not carry conditional subquery \
             branches (out of scope for this feature)",
        ));
    }
    // The subquery must validate as a leaf `AggregateCountOnRange` (which is what the
    // proof descent will ultimately consume).
    ref_validate_leaf_aggregate_count_on_range(subquery)
}

fn ref_validate_leaf_aggregate_sum_on_range(q: &Query) -> Result<&QueryItem, Error> {
    if q.items.len() != 1 {
        return Err(Error::InvalidOperation(
            "AggregateSumOnRange must be the only item in the query",
        ));
    }
    let inner = match &q.items[0] {
        QueryItem::AggregateSumOnRange(inner) => inner.as_ref(),
        _ => {
            return Err(Error::InvalidOperation(
                "validate_aggregate_sum_on_range called on a query without an \
                 AggregateSumOnRange item",
            ));
        }
    };
    match inner {
        QueryItem::Key(_) => {
            return Err(Error::InvalidOperation(
                "AggregateSumOnRange may not wrap Key — use has_raw / get_raw for \
                 existence tests",
            ));
        }
        QueryItem::RangeFull(_) => {
            return Err(Error::InvalidOperation(
                "AggregateSumOnRange may not wrap RangeFull — read the parent \
                 ProvableSumTree element for the unconditional total",
            ));
        }
        QueryItem::AggregateSumOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateSumOnRange may not wrap another AggregateSumOnRange",
            ));
        }
        QueryItem::AggregateCountOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateSumOnRange may not wrap AggregateCountOnRange — the two are \
                 orthogonal aggregate queries",
            ));
        }
        QueryItem::AggregateCountAndSumOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateSumOnRange may not wrap AggregateCountAndSumOnRange — the \
                 aggregate variants are orthogonal",
            ));
        }
        _ => {}
    }
    if q.default_subquery_branch.subquery.is_some()
        || q.default_subquery_branch.subquery_path.is_some()
    {
        return Err(Error::InvalidOperation(
            "AggregateSumOnRange queries may not carry a default subquery branch",
        ));
    }
    if let Some(branches) = &q.conditional_subquery_branches
        && !branches.is_empty()
    {
        return Err(Error::InvalidOperation(
            "AggregateSumOnRange queries may not carry conditional subquery branches",
        ));
    }
    Ok(inner)
}

fn ref_validate_carrier_aggregate_sum_on_range(q: &Query) -> Result<&QueryItem, Error> {
    if q.items.is_empty() {
        return Err(Error::InvalidOperation(
            "carrier AggregateSumOnRange query must have at least one outer item",
        ));
    }
    for item in &q.items {
        match item {
            QueryItem::Key(_)
            | QueryItem::Range(_)
            | QueryItem::RangeInclusive(_)
            | QueryItem::RangeFrom(_)
            | QueryItem::RangeTo(_)
            | QueryItem::RangeToInclusive(_)
            | QueryItem::RangeAfter(_)
            | QueryItem::RangeAfterTo(_)
            | QueryItem::RangeAfterToInclusive(_) => {}
            QueryItem::RangeFull(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateSumOnRange query may not have a RangeFull outer item",
                ));
            }
            QueryItem::AggregateSumOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateSumOnRange query may not own an \
                     AggregateSumOnRange item — use the leaf shape instead",
                ));
            }
            QueryItem::AggregateCountOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateSumOnRange query may not own an \
                     AggregateCountOnRange item — the two aggregate variants are orthogonal",
                ));
            }
            QueryItem::AggregateCountAndSumOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateSumOnRange query may not own an \
                     AggregateCountAndSumOnRange item — the aggregate variants are \
                     orthogonal",
                ));
            }
        }
    }
    let subquery = match q.default_subquery_branch.subquery.as_deref() {
        Some(sub) => sub,
        None => {
            return Err(Error::InvalidOperation(
                "carrier AggregateSumOnRange query must set \
                 default_subquery_branch.subquery to a leaf `AggregateSumOnRange` query",
            ));
        }
    };
    if let Some(path) = &q.default_subquery_branch.subquery_path
        && path.iter().any(|k| k.is_empty())
    {
        return Err(Error::InvalidOperation(
            "carrier AggregateSumOnRange query's subquery_path must contain non-empty keys",
        ));
    }
    if let Some(branches) = &q.conditional_subquery_branches
        && !branches.is_empty()
    {
        return Err(Error::InvalidOperation(
            "carrier AggregateSumOnRange query may not carry conditional subquery \
             branches (out of scope for this feature)",
        ));
    }
    // The subquery must validate as a leaf `AggregateSumOnRange` (which is what the
    // proof descent will ultimately consume).
    ref_validate_leaf_aggregate_sum_on_range(subquery)
}

fn ref_validate_leaf_aggregate_count_and_sum_on_range(q: &Query) -> Result<&QueryItem, Error> {
    if q.items.len() != 1 {
        return Err(Error::InvalidOperation(
            "AggregateCountAndSumOnRange must be the only item in the query",
        ));
    }
    let inner = match &q.items[0] {
        QueryItem::AggregateCountAndSumOnRange(inner) => inner.as_ref(),
        _ => {
            return Err(Error::InvalidOperation(
                "validate_aggregate_count_and_sum_on_range called on a query without an \
                 AggregateCountAndSumOnRange item",
            ));
        }
    };
    match inner {
        QueryItem::Key(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountAndSumOnRange may not wrap Key — use has_raw / get_raw for \
                 existence tests",
            ));
        }
        QueryItem::RangeFull(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountAndSumOnRange may not wrap RangeFull — read the parent \
                 ProvableCountProvableSumTree element for the unconditional totals",
            ));
        }
        QueryItem::AggregateCountAndSumOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountAndSumOnRange may not wrap another \
                 AggregateCountAndSumOnRange",
            ));
        }
        QueryItem::AggregateCountOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountAndSumOnRange may not wrap AggregateCountOnRange — the \
                 aggregate variants are orthogonal",
            ));
        }
        QueryItem::AggregateSumOnRange(_) => {
            return Err(Error::InvalidOperation(
                "AggregateCountAndSumOnRange may not wrap AggregateSumOnRange — the \
                 aggregate variants are orthogonal",
            ));
        }
        _ => {}
    }
    if q.default_subquery_branch.subquery.is_some()
        || q.default_subquery_branch.subquery_path.is_some()
    {
        return Err(Error::InvalidOperation(
            "AggregateCountAndSumOnRange queries may not carry a default subquery branch",
        ));
    }
    if let Some(branches) = &q.conditional_subquery_branches
        && !branches.is_empty()
    {
        return Err(Error::InvalidOperation(
            "AggregateCountAndSumOnRange queries may not carry conditional subquery \
             branches",
        ));
    }
    Ok(inner)
}

fn ref_validate_carrier_aggregate_count_and_sum_on_range(q: &Query) -> Result<&QueryItem, Error> {
    if q.items.is_empty() {
        return Err(Error::InvalidOperation(
            "carrier AggregateCountAndSumOnRange query must have at least one outer item",
        ));
    }
    for item in &q.items {
        match item {
            QueryItem::Key(_)
            | QueryItem::Range(_)
            | QueryItem::RangeInclusive(_)
            | QueryItem::RangeFrom(_)
            | QueryItem::RangeTo(_)
            | QueryItem::RangeToInclusive(_)
            | QueryItem::RangeAfter(_)
            | QueryItem::RangeAfterTo(_)
            | QueryItem::RangeAfterToInclusive(_) => {}
            QueryItem::RangeFull(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateCountAndSumOnRange query may not have a RangeFull \
                     outer item",
                ));
            }
            QueryItem::AggregateCountAndSumOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateCountAndSumOnRange query may not own an \
                     AggregateCountAndSumOnRange item — use the leaf shape instead",
                ));
            }
            QueryItem::AggregateCountOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateCountAndSumOnRange query may not own an \
                     AggregateCountOnRange item — the aggregate variants are orthogonal",
                ));
            }
            QueryItem::AggregateSumOnRange(_) => {
                return Err(Error::InvalidOperation(
                    "carrier AggregateCountAndSumOnRange query may not own an \
                     AggregateSumOnRange item — the aggregate variants are orthogonal",
                ));
            }
        }
    }
    let subquery = match q.default_subquery_branch.subquery.as_deref() {
        Some(sub) => sub,
        None => {
            return Err(Error::InvalidOperation(
                "carrier AggregateCountAndSumOnRange query must set \
                 default_subquery_branch.subquery to a leaf \
                 `AggregateCountAndSumOnRange` query",
            ));
        }
    };
    if let Some(path) = &q.default_subquery_branch.subquery_path
        && path.iter().any(|k| k.is_empty())
    {
        return Err(Error::InvalidOperation(
            "carrier AggregateCountAndSumOnRange query's subquery_path must contain \
             non-empty keys",
        ));
    }
    if let Some(branches) = &q.conditional_subquery_branches
        && !branches.is_empty()
    {
        return Err(Error::InvalidOperation(
            "carrier AggregateCountAndSumOnRange query may not carry conditional \
             subquery branches (out of scope for this feature)",
        ));
    }
    // The subquery must validate as a leaf `AggregateCountAndSumOnRange`.
    ref_validate_leaf_aggregate_count_and_sum_on_range(subquery)
}

// ---- generated inputs ----

fn plain_items() -> Vec<QueryItem> {
    vec![
        QueryItem::Key(b"k".to_vec()),
        QueryItem::Range(b"a".to_vec()..b"m".to_vec()),
        QueryItem::RangeInclusive(b"a".to_vec()..=b"m".to_vec()),
        QueryItem::RangeFull(..),
        QueryItem::RangeFrom(b"c".to_vec()..),
        QueryItem::RangeTo(..b"x".to_vec()),
        QueryItem::RangeToInclusive(..=b"x".to_vec()),
        QueryItem::RangeAfter(b"c".to_vec()..),
        QueryItem::RangeAfterTo(b"c".to_vec()..b"x".to_vec()),
        QueryItem::RangeAfterToInclusive(b"c".to_vec()..=b"x".to_vec()),
    ]
}

fn aggregate_items(inner: &QueryItem) -> Vec<QueryItem> {
    vec![
        QueryItem::AggregateCountOnRange(Box::new(inner.clone())),
        QueryItem::AggregateSumOnRange(Box::new(inner.clone())),
        QueryItem::AggregateCountAndSumOnRange(Box::new(inner.clone())),
    ]
}

/// Every plain item, and every aggregate wrapping a plain item or an
/// aggregate of a range.
fn single_items() -> Vec<QueryItem> {
    let mut inners = plain_items();
    inners.extend(aggregate_items(&QueryItem::Range(
        b"a".to_vec()..b"m".to_vec(),
    )));
    let mut items = plain_items();
    for inner in &inners {
        items.extend(aggregate_items(inner));
    }
    items
}

/// Item lists: empty, every single item, and every pair of plain items and
/// aggregates of a range.
fn item_lists() -> Vec<Vec<QueryItem>> {
    let mut pool = plain_items();
    pool.extend(aggregate_items(&QueryItem::Range(
        b"a".to_vec()..b"m".to_vec(),
    )));
    let mut lists = vec![vec![]];
    for item in single_items() {
        lists.push(vec![item]);
    }
    for a in &pool {
        for b in &pool {
            lists.push(vec![a.clone(), b.clone()]);
        }
    }
    lists
}

fn leaf_query(item: QueryItem) -> Query {
    Query {
        items: vec![item],
        ..Query::new()
    }
}

fn subqueries() -> Vec<Option<Box<Query>>> {
    let range = QueryItem::Range(b"a".to_vec()..b"m".to_vec());
    let mut subs = vec![None, Some(Box::new(Query::new()))];
    for item in aggregate_items(&range) {
        subs.push(Some(Box::new(leaf_query(item))));
    }
    for item in aggregate_items(&QueryItem::Key(b"k".to_vec())) {
        subs.push(Some(Box::new(leaf_query(item))));
    }
    // A leaf that itself carries a subquery path.
    let mut with_path = leaf_query(QueryItem::AggregateCountOnRange(Box::new(range)));
    with_path.default_subquery_branch.subquery_path = Some(vec![b"p".to_vec()]);
    subs.push(Some(Box::new(with_path)));
    subs
}

fn subquery_paths() -> Vec<Option<Vec<Vec<u8>>>> {
    vec![
        None,
        Some(vec![]),
        Some(vec![b"p".to_vec()]),
        Some(vec![b"p".to_vec(), b"".to_vec()]),
        Some(vec![b"".to_vec(), b"p".to_vec()]),
        Some(vec![b"".to_vec()]),
    ]
}

fn conditional_branches() -> Vec<Option<IndexMap<QueryItem, SubqueryBranch>>> {
    let mut one = IndexMap::new();
    one.insert(QueryItem::Key(b"c".to_vec()), SubqueryBranch::default());
    vec![None, Some(IndexMap::new()), Some(one)]
}

fn queries() -> Vec<Query> {
    let mut queries = Vec::new();
    for items in item_lists() {
        for subquery in subqueries() {
            for subquery_path in subquery_paths() {
                for conditional in conditional_branches() {
                    queries.push(Query {
                        items: items.clone(),
                        default_subquery_branch: SubqueryBranch {
                            subquery_path: subquery_path.clone(),
                            subquery: subquery.clone(),
                        },
                        conditional_subquery_branches: conditional,
                        ..Query::new()
                    });
                }
            }
        }
    }
    queries
}

/// The same borrowed item, or an error of the same variant and message.
fn assert_same<'a>(
    q: &Query,
    got: Result<&'a QueryItem, Error>,
    want: Result<&'a QueryItem, Error>,
) {
    match (got, want) {
        (Ok(a), Ok(b)) => assert!(std::ptr::eq(a, b), "different item for {q:?}"),
        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "for {q:?}"),
        (a, b) => panic!("{a:?} != {b:?} for {q:?}"),
    }
}

#[test]
fn aggregate_validators_match_reference() {
    let queries = queries();
    assert!(queries.len() > 30_000);
    for q in &queries {
        assert_same(
            q,
            q.validate_leaf_aggregate_count_on_range(),
            ref_validate_leaf_aggregate_count_on_range(q),
        );
        assert_same(
            q,
            q.validate_carrier_aggregate_count_on_range(),
            ref_validate_carrier_aggregate_count_on_range(q),
        );
        assert_same(
            q,
            q.validate_leaf_aggregate_sum_on_range(),
            ref_validate_leaf_aggregate_sum_on_range(q),
        );
        assert_same(
            q,
            q.validate_carrier_aggregate_sum_on_range(),
            ref_validate_carrier_aggregate_sum_on_range(q),
        );
        assert_same(
            q,
            q.validate_leaf_aggregate_count_and_sum_on_range(),
            ref_validate_leaf_aggregate_count_and_sum_on_range(q),
        );
        assert_same(
            q,
            q.validate_carrier_aggregate_count_and_sum_on_range(),
            ref_validate_carrier_aggregate_count_and_sum_on_range(q),
        );
    }
}

#[test]
fn any_empty_key_matches_iterator_any() {
    let keys: [&[u8]; 3] = [b"", b"a", b"bc"];
    let mut lists: Vec<Vec<Vec<u8>>> = vec![vec![]];
    for _ in 0..3 {
        let mut longer = Vec::new();
        for list in &lists {
            for key in keys {
                let mut l = list.clone();
                l.push(key.to_vec());
                longer.push(l);
            }
        }
        lists.extend(longer);
    }
    for list in &lists {
        assert_eq!(
            any_empty_key(list),
            list.iter().any(|k| k.is_empty()),
            "{list:?}"
        );
    }
}
