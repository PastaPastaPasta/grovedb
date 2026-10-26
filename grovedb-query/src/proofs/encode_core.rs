//! Pure encoded-length computation for proof [`Op`]s.
//!
//! [`op_encoding_length`] returns exactly what
//! `<Op as ed::Encode>::encoding_length` returns, without going through
//! `ed`, `std::io` or `integer-encoding` so that Charon/Aeneas can translate
//! it. The decoder uses it to advance past an operator.
//!
//! The computation mirrors `encode_into`'s choice of layout: a value shorter
//! than 65536 bytes uses the 4-byte header (opcode, key length, `u16` value
//! length), anything else the 6-byte header (`u32` value length). Like
//! `ed`'s version it does not check that keys fit in 255 bytes or that values
//! are within the decoder's maximum length; `encode_into` is what rejects
//! over-long keys.

use super::{Node, Op, TreeFeatureType, HASH_LENGTH};

/// Number of bytes a `u64` takes as a LEB128-style varint
/// (`integer-encoding`'s `required_encoded_space_unsigned`).
pub fn varint_len_u64(value: u64) -> usize {
    let mut v = value;
    let mut count: usize = 0;
    while v > 0 {
        count += 1;
        v >>= 7;
    }
    if count == 0 {
        1
    } else {
        count
    }
}

/// Number of bytes an `i64` takes as a zigzag varint
/// (`integer-encoding`'s `required_encoded_space_signed`).
pub fn varint_len_i64(value: i64) -> usize {
    // zigzag: ((v << 1) ^ (v >> 63)) as u64
    let bits = value as u64;
    let sign_mask = if value < 0 { u64::MAX } else { 0 };
    varint_len_u64((bits << 1) ^ sign_mask)
}

/// Encoded length of a [`TreeFeatureType`] (tag byte plus payload).
pub fn tree_feature_type_encoding_length(feature_type: &TreeFeatureType) -> usize {
    match feature_type {
        TreeFeatureType::BasicMerkNode => 1,
        TreeFeatureType::SummedMerkNode(sum) => 1 + varint_len_i64(*sum),
        TreeFeatureType::BigSummedMerkNode(_) => 17,
        TreeFeatureType::CountedMerkNode(count) => 1 + varint_len_u64(*count),
        TreeFeatureType::CountedSummedMerkNode(count, sum) => {
            1 + varint_len_u64(*count) + varint_len_i64(*sum)
        }
        TreeFeatureType::ProvableCountedMerkNode(count) => 1 + varint_len_u64(*count),
        TreeFeatureType::ProvableCountedSummedMerkNode(count, sum) => {
            1 + varint_len_u64(*count) + varint_len_i64(*sum)
        }
        TreeFeatureType::ProvableSummedMerkNode(sum) => 1 + varint_len_i64(*sum),
        TreeFeatureType::ProvableCountedAndProvableSummedMerkNode(count, sum) => {
            1 + varint_len_u64(*count) + varint_len_i64(*sum)
        }
    }
}

/// Opcode, key length byte, value length (2 or 4 bytes) plus the key and
/// value bytes themselves.
fn key_value_length(key: &[u8], value: &[u8]) -> usize {
    let header: usize = if value.len() < 65536 { 4 } else { 6 };
    header + key.len() + value.len()
}

/// Encoded length of a [`Node`] inside a `Push`/`PushInverted` (both
/// directions use the same layout).
pub fn node_encoding_length(node: &Node) -> usize {
    match node {
        Node::Hash(_) => 1 + HASH_LENGTH,
        Node::KVHash(_) => 1 + HASH_LENGTH,
        Node::KVDigest(key, _) => 2 + key.len() + HASH_LENGTH,
        Node::KV(key, value) => key_value_length(key, value),
        Node::KVValueHash(key, value, _) => key_value_length(key, value) + HASH_LENGTH,
        Node::KVRefValueHash(key, value, _) => key_value_length(key, value) + HASH_LENGTH,
        Node::KVBackwardsReferencesValueHash(key, value, _) => {
            key_value_length(key, value) + HASH_LENGTH
        }
        Node::KVValueHashFeatureType(key, value, _, feature_type) => {
            key_value_length(key, value)
                + HASH_LENGTH
                + tree_feature_type_encoding_length(feature_type)
        }
        Node::KVCount(key, value, _) => key_value_length(key, value) + 8,
        Node::KVHashCount(_, _) => 1 + HASH_LENGTH + 8,
        Node::KVRefValueHashCount(key, value, _, _) => {
            key_value_length(key, value) + HASH_LENGTH + 8
        }
        Node::KVDigestCount(key, _, _) => 2 + key.len() + HASH_LENGTH + 8,
        Node::HashWithCount(_, _, _, _) => 1 + 3 * HASH_LENGTH + 8,
        Node::KVValueHashFeatureTypeWithChildHash(key, value, _, feature_type, _) => {
            key_value_length(key, value)
                + HASH_LENGTH
                + tree_feature_type_encoding_length(feature_type)
                + HASH_LENGTH
        }
        Node::KVSum(key, value, _) => key_value_length(key, value) + 8,
        Node::KVHashSum(_, _) => 1 + HASH_LENGTH + 8,
        Node::KVRefValueHashSum(key, value, _, _) => key_value_length(key, value) + HASH_LENGTH + 8,
        Node::KVDigestSum(key, _, _) => 2 + key.len() + HASH_LENGTH + 8,
        Node::HashWithSum(_, _, _, _) => 1 + 3 * HASH_LENGTH + 8,
        Node::KVCountSum(key, value, _, _) => key_value_length(key, value) + 16,
        Node::KVHashCountSum(_, _, _) => 1 + HASH_LENGTH + 16,
        Node::KVRefValueHashCountSum(key, value, _, _, _) => {
            key_value_length(key, value) + HASH_LENGTH + 16
        }
        Node::KVDigestCountSum(key, _, _, _) => 2 + key.len() + HASH_LENGTH + 16,
        Node::HashWithCountAndSum(_, _, _, _, _) => 1 + 3 * HASH_LENGTH + 16,
    }
}

/// Number of bytes `op` occupies when encoded; identical to
/// `<Op as ed::Encode>::encoding_length(op)`, which never fails.
pub fn op_encoding_length(op: &Op) -> usize {
    match op {
        Op::Push(node) => node_encoding_length(node),
        Op::PushInverted(node) => node_encoding_length(node),
        Op::Parent => 1,
        Op::Child => 1,
        Op::ParentInverted => 1,
        Op::ChildInverted => 1,
    }
}

#[cfg(test)]
mod tests {
    use ed::Encode;
    use integer_encoding::VarInt;

    use super::*;

    fn old(op: &Op) -> usize {
        <Op as Encode>::encoding_length(op).expect("ed encoding_length")
    }

    fn hash(seed: u8) -> [u8; HASH_LENGTH] {
        [seed; HASH_LENGTH]
    }

    fn feature_types() -> Vec<TreeFeatureType> {
        use TreeFeatureType::*;
        let mut v = vec![BasicMerkNode, BigSummedMerkNode(i128::MIN)];
        let counts = [
            0u64,
            1,
            127,
            128,
            16383,
            16384,
            (1 << 56) - 1,
            1 << 56,
            1 << 63,
            u64::MAX,
        ];
        let sums = [
            0i64,
            1,
            -1,
            63,
            -64,
            64,
            -65,
            8191,
            -8192,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
        ];
        for s in sums {
            v.push(SummedMerkNode(s));
            v.push(ProvableSummedMerkNode(s));
        }
        for c in counts {
            v.push(CountedMerkNode(c));
            v.push(ProvableCountedMerkNode(c));
            for s in sums {
                v.push(CountedSummedMerkNode(c, s));
                v.push(ProvableCountedSummedMerkNode(c, s));
                v.push(ProvableCountedAndProvableSummedMerkNode(c, s));
            }
        }
        v
    }

    fn all_nodes(key: &[u8], value: &[u8]) -> Vec<Node> {
        let k = key.to_vec();
        let v = value.to_vec();
        let mut nodes = vec![
            Node::Hash(hash(1)),
            Node::KVHash(hash(2)),
            Node::KVDigest(k.clone(), hash(3)),
            Node::KV(k.clone(), v.clone()),
            Node::KVValueHash(k.clone(), v.clone(), hash(4)),
            Node::KVRefValueHash(k.clone(), v.clone(), hash(5)),
            Node::KVBackwardsReferencesValueHash(k.clone(), v.clone(), hash(6)),
            Node::KVCount(k.clone(), v.clone(), 7),
            Node::KVHashCount(hash(7), u64::MAX),
            Node::KVRefValueHashCount(k.clone(), v.clone(), hash(8), 9),
            Node::KVDigestCount(k.clone(), hash(9), 10),
            Node::HashWithCount(hash(10), hash(11), hash(12), 13),
            Node::KVSum(k.clone(), v.clone(), -14),
            Node::KVHashSum(hash(13), i64::MIN),
            Node::KVRefValueHashSum(k.clone(), v.clone(), hash(14), i64::MAX),
            Node::KVDigestSum(k.clone(), hash(15), -16),
            Node::HashWithSum(hash(16), hash(17), hash(18), 19),
            Node::KVCountSum(k.clone(), v.clone(), 20, -21),
            Node::KVHashCountSum(hash(19), 22, -23),
            Node::KVRefValueHashCountSum(k.clone(), v.clone(), hash(20), 24, -25),
            Node::KVDigestCountSum(k.clone(), hash(21), 26, -27),
            Node::HashWithCountAndSum(hash(22), hash(23), hash(24), 28, -29),
        ];
        for ft in feature_types().into_iter().step_by(7) {
            nodes.push(Node::KVValueHashFeatureType(
                k.clone(),
                v.clone(),
                hash(30),
                ft,
            ));
            nodes.push(Node::KVValueHashFeatureTypeWithChildHash(
                k.clone(),
                v.clone(),
                hash(31),
                ft,
                hash(32),
            ));
        }
        nodes
    }

    fn ops_for(key: &[u8], value: &[u8]) -> Vec<Op> {
        let mut ops = vec![Op::Parent, Op::Child, Op::ParentInverted, Op::ChildInverted];
        for node in all_nodes(key, value) {
            ops.push(Op::Push(node.clone()));
            ops.push(Op::PushInverted(node));
        }
        ops
    }

    #[test]
    fn varint_lengths_match_integer_encoding() {
        let mut values: Vec<u64> = vec![0, 1, u64::MAX, u64::MAX - 1];
        for shift in 0..64 {
            let p = 1u64 << shift;
            values.extend_from_slice(&[p - 1, p, p.wrapping_add(1)]);
        }
        for v in values {
            assert_eq!(varint_len_u64(v), v.required_space(), "u64 {v}");
            let s = v as i64;
            assert_eq!(varint_len_i64(s), s.required_space(), "i64 {s}");
        }
        for s in [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX, i64::MAX - 1] {
            assert_eq!(varint_len_i64(s), s.required_space(), "i64 {s}");
        }
        let mut x: u64 = 0x1234_5678_9abc_def1;
        for _ in 0..100_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let shifted = x >> (x % 64);
            assert_eq!(varint_len_u64(shifted), shifted.required_space());
            assert_eq!(
                varint_len_i64(shifted as i64),
                (shifted as i64).required_space()
            );
        }
    }

    #[test]
    fn feature_type_lengths_match_ed() {
        for ft in feature_types() {
            assert_eq!(
                tree_feature_type_encoding_length(&ft),
                Encode::encoding_length(&ft).unwrap(),
                "{ft:?}"
            );
        }
    }

    #[test]
    fn every_variant_matches_ed_across_length_boundaries() {
        // Value lengths around the u16/u32 header switch and key lengths
        // around the u8 limit (over-long keys never reach `encode_into`, but
        // `ed`'s encoding_length still answers for them).
        let value_lens = [0usize, 1, 255, 256, 65534, 65535, 65536, 65537, 70_000];
        let key_lens = [0usize, 1, 2, 254, 255, 256, 300];
        for &vl in &value_lens {
            let value = vec![0xabu8; vl];
            for &kl in &key_lens {
                if vl > 1000 && kl > 2 && kl != 255 {
                    continue;
                }
                let key = vec![0xcdu8; kl];
                for op in ops_for(&key, &value) {
                    assert_eq!(op_encoding_length(&op), old(&op), "k={kl} v={vl} {op:?}");
                }
            }
        }
    }

    #[test]
    fn matches_ed_beyond_max_value_len() {
        // MAX_VALUE_LEN + 1 bytes would be rejected by the decoder, but the
        // length computation must still agree. One 64 MiB buffer, moved
        // between the two directions rather than cloned.
        let value = vec![1u8; crate::proofs::decode_core::MAX_VALUE_LEN as usize + 1];
        let push = Op::Push(Node::KVSum(vec![1], value, 5));
        assert_eq!(op_encoding_length(&push), old(&push));
        let Op::Push(node) = push else { unreachable!() };
        let inverted = Op::PushInverted(node);
        assert_eq!(op_encoding_length(&inverted), old(&inverted));
    }

    #[test]
    fn matches_actual_encoded_size_for_encodable_ops() {
        for op in ops_for(&[1, 2, 3], &[4, 5])
            .into_iter()
            .chain(ops_for(&[7u8; 255], &vec![1u8; 65536]))
        {
            let mut bytes = vec![];
            op.encode_into(&mut bytes).unwrap();
            assert_eq!(op_encoding_length(&op), bytes.len(), "{op:?}");
        }
    }

    #[test]
    fn matches_ed_on_decoded_random_inputs() {
        use crate::proofs::decode_core::decode_op;
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let mut decoded = 0;
        for i in 0..60_000 {
            let len = (next() % 150) as usize;
            let mut bytes: Vec<u8> = (0..len)
                .map(|_| {
                    let r = next();
                    if r & 3 == 0 {
                        (r >> 8) as u8
                    } else {
                        (r >> 8) as u8 & 0x0f
                    }
                })
                .collect();
            if i % 2 == 0 && !bytes.is_empty() {
                // Bias towards valid opcodes so most inputs decode.
                let tags = [
                    0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0e, 0x14, 0x15, 0x18, 0x1a, 0x1c,
                    0x1e, 0x20, 0x23, 0x30, 0x32, 0x36, 0x40, 0x42, 0x43, 0x46, 0x50, 0x10,
                ];
                bytes[0] = tags[(next() as usize) % tags.len()];
            }
            if let Ok(op) = decode_op(&bytes) {
                decoded += 1;
                assert_eq!(op_encoding_length(&op), old(&op), "{op:?}");
            }
        }
        assert!(decoded > 5_000, "only {decoded} random inputs decoded");
    }
}
