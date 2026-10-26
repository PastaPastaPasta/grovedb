//! Pure, slice-based decoding core for proof [`Op`]s.
//!
//! This module decodes the same wire format as `impl ed::Decode for Op`
//! (see `encoding.rs`) but works directly on a byte slice through a small
//! cursor instead of going through `std::io::Read`. It deliberately avoids
//! generics over `Read`, trait objects, closures, iterator adaptors and
//! `std::io` so that it can be translated by Charon/Aeneas.
//!
//! The accept/reject behavior and the decoded values are identical to the
//! `ed::Decode` implementation; the differential tests at the bottom of this
//! file check that.

use super::{Node, Op, TreeFeatureType, HASH_LENGTH};

/// Maximum allowed value length for large value variants (64MB).
/// This prevents DoS attacks via malicious proofs specifying unreasonably large
/// allocations.
pub(crate) const MAX_VALUE_LEN: u32 = 64 * 1024 * 1024;

/// Why decoding an [`Op`] from a byte slice failed.
///
/// Each variant corresponds to exactly one of the failure modes of the
/// `ed::Decode` based decoder, so callers can reproduce its error strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpDecodeError {
    /// An unknown opcode / feature-type tag, or a value length over the
    /// allowed maximum (reported with the opcode), as `ed::Error::UnexpectedByte`.
    UnexpectedByte(u8),
    /// A fixed-size read ran past the end of the input
    /// (`read_exact` EOF: "failed to fill whole buffer").
    UnexpectedEof,
    /// A varint was empty or ended before a valid terminating byte
    /// (`integer-encoding` "Reached EOF").
    TruncatedVarint,
    /// A varint had more continuation bytes than fit in 64 bits
    /// (`integer-encoding` "Unterminated varint").
    UnterminatedVarint,
}

/// A read cursor over a byte slice.
///
/// Invariant: `pos <= bytes.len()`.
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Create a cursor positioned at the start of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    /// Number of bytes consumed so far.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Number of bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    /// Read one byte.
    pub fn read_u8(&mut self) -> Result<u8, OpDecodeError> {
        if self.pos >= self.bytes.len() {
            return Err(OpDecodeError::UnexpectedEof);
        }
        let b = self.bytes[self.pos];
        self.pos += 1;
        Ok(b)
    }

    /// Read a big-endian `u16`.
    pub fn read_u16_be(&mut self) -> Result<u16, OpDecodeError> {
        if self.remaining() < 2 {
            return Err(OpDecodeError::UnexpectedEof);
        }
        let hi = self.read_u8()?;
        let lo = self.read_u8()?;
        Ok((u16::from(hi) << 8) | u16::from(lo))
    }

    /// Read a big-endian `u32`.
    pub fn read_u32_be(&mut self) -> Result<u32, OpDecodeError> {
        if self.remaining() < 4 {
            return Err(OpDecodeError::UnexpectedEof);
        }
        let mut v: u32 = 0;
        let mut i = 0;
        while i < 4 {
            let b = self.read_u8()?;
            v = (v << 8) | u32::from(b);
            i += 1;
        }
        Ok(v)
    }

    /// Read a big-endian `u64`.
    pub fn read_u64_be(&mut self) -> Result<u64, OpDecodeError> {
        if self.remaining() < 8 {
            return Err(OpDecodeError::UnexpectedEof);
        }
        let mut v: u64 = 0;
        let mut i = 0;
        while i < 8 {
            let b = self.read_u8()?;
            v = (v << 8) | u64::from(b);
            i += 1;
        }
        Ok(v)
    }

    /// Read a big-endian `i64` (two's complement).
    pub fn read_i64_be(&mut self) -> Result<i64, OpDecodeError> {
        let v = self.read_u64_be()?;
        Ok(v as i64)
    }

    /// Read a big-endian `i128` (two's complement).
    pub fn read_i128_be(&mut self) -> Result<i128, OpDecodeError> {
        if self.remaining() < 16 {
            return Err(OpDecodeError::UnexpectedEof);
        }
        let mut v: u128 = 0;
        let mut i = 0;
        while i < 16 {
            let b = self.read_u8()?;
            v = (v << 8) | u128::from(b);
            i += 1;
        }
        Ok(v as i128)
    }

    /// Read exactly `len` bytes into a new vector.
    pub fn read_vec(&mut self, len: usize) -> Result<Vec<u8>, OpDecodeError> {
        if self.remaining() < len {
            return Err(OpDecodeError::UnexpectedEof);
        }
        let end = self.pos + len;
        let v = self.bytes[self.pos..end].to_vec();
        self.pos = end;
        Ok(v)
    }

    /// Read a 32-byte hash.
    pub fn read_hash(&mut self) -> Result<[u8; HASH_LENGTH], OpDecodeError> {
        if self.remaining() < HASH_LENGTH {
            return Err(OpDecodeError::UnexpectedEof);
        }
        let end = self.pos + HASH_LENGTH;
        let mut out = [0u8; HASH_LENGTH];
        out.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    /// Read a key: one length byte followed by that many bytes.
    pub fn read_key(&mut self) -> Result<Vec<u8>, OpDecodeError> {
        let len = self.read_u8()?;
        self.read_vec(usize::from(len))
    }

    /// Read a value: a big-endian `u16` length (`large == false`) or `u32`
    /// length (`large == true`, rejected with `UnexpectedByte(tag)` when over
    /// [`MAX_VALUE_LEN`]) followed by that many bytes.
    pub fn read_value(&mut self, large: bool, tag: u8) -> Result<Vec<u8>, OpDecodeError> {
        let len = if large {
            let l = self.read_u32_be()?;
            if l > MAX_VALUE_LEN {
                return Err(OpDecodeError::UnexpectedByte(tag));
            }
            l as usize
        } else {
            usize::from(self.read_u16_be()?)
        };
        self.read_vec(len)
    }

    /// Read a LEB128-style varint exactly as `integer-encoding` 4.1.0's
    /// `VarIntReader::read_varint::<u64>` does, returning the raw `u64`
    /// that `u64::decode_var` produces.
    ///
    /// At most 10 bytes are accepted. The 10th byte must be 0 or 1 for the
    /// value to be representable.
    pub fn read_varint_u64(&mut self) -> Result<u64, OpDecodeError> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        let mut count: usize = 0;
        let mut last: u8 = 0;
        loop {
            // The varint is complete once a byte without the continuation
            // bit has been pushed.
            if count > 0 && (last & 0x80) == 0 {
                break;
            }
            // Out of input.
            if self.pos >= self.bytes.len() {
                if count == 0 {
                    return Err(OpDecodeError::TruncatedVarint);
                }
                break;
            }
            let b = self.bytes[self.pos];
            self.pos += 1;
            // More bytes than a u64 varint can have.
            if count >= 10 {
                return Err(OpDecodeError::UnterminatedVarint);
            }
            result |= u64::from(b & 0x7f) << shift;
            shift += 7;
            last = b;
            count += 1;
        }
        // Decode succeeds only if the terminating byte (without continuation
        // bit) was reached, and a 10th byte is at most 1.
        if count == 10 {
            if last >= 2 {
                return Err(OpDecodeError::TruncatedVarint);
            }
        } else if (last & 0x80) != 0 {
            return Err(OpDecodeError::TruncatedVarint);
        }
        Ok(result)
    }

    /// Read a zigzag varint `i64` exactly as `integer-encoding` 4.1.0's
    /// `VarIntReader::read_varint::<i64>` does.
    pub fn read_varint_i64(&mut self) -> Result<i64, OpDecodeError> {
        let raw = self.read_varint_u64()?;
        let half = raw >> 1;
        if (raw & 1) == 0 {
            Ok(half as i64)
        } else {
            Ok((!half) as i64)
        }
    }
}

/// Decode a [`TreeFeatureType`] (feature-type tag byte followed by its
/// payload) from the cursor.
pub fn decode_tree_feature_type(cur: &mut Cursor<'_>) -> Result<TreeFeatureType, OpDecodeError> {
    let tag = cur.read_u8()?;
    match tag {
        0 => Ok(TreeFeatureType::BasicMerkNode),
        1 => {
            let sum = cur.read_varint_i64()?;
            Ok(TreeFeatureType::SummedMerkNode(sum))
        }
        2 => {
            let sum = cur.read_i128_be()?;
            Ok(TreeFeatureType::BigSummedMerkNode(sum))
        }
        3 => {
            let count = cur.read_varint_u64()?;
            Ok(TreeFeatureType::CountedMerkNode(count))
        }
        4 => {
            let count = cur.read_varint_u64()?;
            let sum = cur.read_varint_i64()?;
            Ok(TreeFeatureType::CountedSummedMerkNode(count, sum))
        }
        5 => {
            let count = cur.read_varint_u64()?;
            Ok(TreeFeatureType::ProvableCountedMerkNode(count))
        }
        6 => {
            let count = cur.read_varint_u64()?;
            let sum = cur.read_varint_i64()?;
            Ok(TreeFeatureType::ProvableCountedSummedMerkNode(count, sum))
        }
        7 => {
            let sum = cur.read_varint_i64()?;
            Ok(TreeFeatureType::ProvableSummedMerkNode(sum))
        }
        8 => {
            let count = cur.read_varint_u64()?;
            let sum = cur.read_varint_i64()?;
            Ok(TreeFeatureType::ProvableCountedAndProvableSummedMerkNode(
                count, sum,
            ))
        }
        other => Err(OpDecodeError::UnexpectedByte(other)),
    }
}

/// The [`Node`] variant an opcode carries (ignoring push direction and the
/// small/large value-length encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeKind {
    Hash,
    KVHash,
    KV,
    KVValueHash,
    KVDigest,
    KVRefValueHash,
    KVBackwardsReferencesValueHash,
    KVValueHashFeatureType,
    KVCount,
    KVHashCount,
    KVRefValueHashCount,
    KVDigestCount,
    KVValueHashFeatureTypeWithChildHash,
    HashWithCount,
    KVSum,
    KVHashSum,
    KVRefValueHashSum,
    KVDigestSum,
    HashWithSum,
    KVCountSum,
    KVHashCountSum,
    KVRefValueHashCountSum,
    KVDigestCountSum,
    HashWithCountAndSum,
}

/// What a `Push`/`PushInverted` opcode byte means.
#[derive(Clone, Copy, Debug)]
struct PushTag {
    inverted: bool,
    kind: NodeKind,
    /// The value length is a `u32` (instead of a `u16`).
    large: bool,
}

const fn push_tag(inverted: bool, kind: NodeKind, large: bool) -> PushTag {
    PushTag {
        inverted,
        kind,
        large,
    }
}

/// Classify a push opcode byte; `None` for every byte that is not a push
/// opcode (including the parent/child opcodes, which are handled separately).
fn classify_push_tag(tag: u8) -> Option<PushTag> {
    match tag {
        // Push
        0x01 => Some(push_tag(false, NodeKind::Hash, false)),
        0x02 => Some(push_tag(false, NodeKind::KVHash, false)),
        0x03 => Some(push_tag(false, NodeKind::KV, false)),
        0x20 => Some(push_tag(false, NodeKind::KV, true)),
        0x04 => Some(push_tag(false, NodeKind::KVValueHash, false)),
        0x21 => Some(push_tag(false, NodeKind::KVValueHash, true)),
        0x05 => Some(push_tag(false, NodeKind::KVDigest, false)),
        0x06 => Some(push_tag(false, NodeKind::KVRefValueHash, false)),
        0x22 => Some(push_tag(false, NodeKind::KVRefValueHash, true)),
        0x50 => Some(push_tag(
            false,
            NodeKind::KVBackwardsReferencesValueHash,
            false,
        )),
        0x51 => Some(push_tag(
            false,
            NodeKind::KVBackwardsReferencesValueHash,
            true,
        )),
        0x07 => Some(push_tag(false, NodeKind::KVValueHashFeatureType, false)),
        0x23 => Some(push_tag(false, NodeKind::KVValueHashFeatureType, true)),
        0x14 => Some(push_tag(false, NodeKind::KVCount, false)),
        0x24 => Some(push_tag(false, NodeKind::KVCount, true)),
        0x15 => Some(push_tag(false, NodeKind::KVHashCount, false)),
        0x18 => Some(push_tag(false, NodeKind::KVRefValueHashCount, false)),
        0x25 => Some(push_tag(false, NodeKind::KVRefValueHashCount, true)),
        0x1a => Some(push_tag(false, NodeKind::KVDigestCount, false)),
        0x1c => Some(push_tag(
            false,
            NodeKind::KVValueHashFeatureTypeWithChildHash,
            false,
        )),
        0x2e => Some(push_tag(
            false,
            NodeKind::KVValueHashFeatureTypeWithChildHash,
            true,
        )),
        0x1e => Some(push_tag(false, NodeKind::HashWithCount, false)),
        0x30 => Some(push_tag(false, NodeKind::KVSum, false)),
        0x31 => Some(push_tag(false, NodeKind::KVSum, true)),
        0x32 => Some(push_tag(false, NodeKind::KVHashSum, false)),
        0x33 => Some(push_tag(false, NodeKind::KVRefValueHashSum, false)),
        0x34 => Some(push_tag(false, NodeKind::KVRefValueHashSum, true)),
        0x35 => Some(push_tag(false, NodeKind::KVDigestSum, false)),
        0x36 => Some(push_tag(false, NodeKind::HashWithSum, false)),
        0x40 => Some(push_tag(false, NodeKind::KVCountSum, false)),
        0x41 => Some(push_tag(false, NodeKind::KVCountSum, true)),
        0x42 => Some(push_tag(false, NodeKind::KVHashCountSum, false)),
        0x43 => Some(push_tag(false, NodeKind::KVRefValueHashCountSum, false)),
        0x44 => Some(push_tag(false, NodeKind::KVRefValueHashCountSum, true)),
        0x45 => Some(push_tag(false, NodeKind::KVDigestCountSum, false)),
        0x46 => Some(push_tag(false, NodeKind::HashWithCountAndSum, false)),
        // PushInverted
        0x08 => Some(push_tag(true, NodeKind::Hash, false)),
        0x09 => Some(push_tag(true, NodeKind::KVHash, false)),
        0x0a => Some(push_tag(true, NodeKind::KV, false)),
        0x28 => Some(push_tag(true, NodeKind::KV, true)),
        0x0b => Some(push_tag(true, NodeKind::KVValueHash, false)),
        0x29 => Some(push_tag(true, NodeKind::KVValueHash, true)),
        0x0c => Some(push_tag(true, NodeKind::KVDigest, false)),
        0x0d => Some(push_tag(true, NodeKind::KVRefValueHash, false)),
        0x2a => Some(push_tag(true, NodeKind::KVRefValueHash, true)),
        0x52 => Some(push_tag(
            true,
            NodeKind::KVBackwardsReferencesValueHash,
            false,
        )),
        0x53 => Some(push_tag(
            true,
            NodeKind::KVBackwardsReferencesValueHash,
            true,
        )),
        0x0e => Some(push_tag(true, NodeKind::KVValueHashFeatureType, false)),
        0x2b => Some(push_tag(true, NodeKind::KVValueHashFeatureType, true)),
        0x16 => Some(push_tag(true, NodeKind::KVCount, false)),
        0x2c => Some(push_tag(true, NodeKind::KVCount, true)),
        0x17 => Some(push_tag(true, NodeKind::KVHashCount, false)),
        0x19 => Some(push_tag(true, NodeKind::KVRefValueHashCount, false)),
        0x2d => Some(push_tag(true, NodeKind::KVRefValueHashCount, true)),
        0x1b => Some(push_tag(true, NodeKind::KVDigestCount, false)),
        0x1d => Some(push_tag(
            true,
            NodeKind::KVValueHashFeatureTypeWithChildHash,
            false,
        )),
        0x2f => Some(push_tag(
            true,
            NodeKind::KVValueHashFeatureTypeWithChildHash,
            true,
        )),
        0x1f => Some(push_tag(true, NodeKind::HashWithCount, false)),
        0x37 => Some(push_tag(true, NodeKind::KVSum, false)),
        0x38 => Some(push_tag(true, NodeKind::KVSum, true)),
        0x39 => Some(push_tag(true, NodeKind::KVHashSum, false)),
        0x3a => Some(push_tag(true, NodeKind::KVRefValueHashSum, false)),
        0x3b => Some(push_tag(true, NodeKind::KVRefValueHashSum, true)),
        0x3c => Some(push_tag(true, NodeKind::KVDigestSum, false)),
        0x3d => Some(push_tag(true, NodeKind::HashWithSum, false)),
        0x47 => Some(push_tag(true, NodeKind::KVCountSum, false)),
        0x48 => Some(push_tag(true, NodeKind::KVCountSum, true)),
        0x49 => Some(push_tag(true, NodeKind::KVHashCountSum, false)),
        0x4a => Some(push_tag(true, NodeKind::KVRefValueHashCountSum, false)),
        0x4b => Some(push_tag(true, NodeKind::KVRefValueHashCountSum, true)),
        0x4c => Some(push_tag(true, NodeKind::KVDigestCountSum, false)),
        0x4d => Some(push_tag(true, NodeKind::HashWithCountAndSum, false)),
        _ => None,
    }
}

/// Decode the payload of a push opcode (everything after the opcode byte).
fn decode_node(
    cur: &mut Cursor<'_>,
    kind: NodeKind,
    large: bool,
    tag: u8,
) -> Result<Node, OpDecodeError> {
    match kind {
        NodeKind::Hash => {
            let hash = cur.read_hash()?;
            Ok(Node::Hash(hash))
        }
        NodeKind::KVHash => {
            let hash = cur.read_hash()?;
            Ok(Node::KVHash(hash))
        }
        NodeKind::KV => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            Ok(Node::KV(key, value))
        }
        NodeKind::KVValueHash => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let value_hash = cur.read_hash()?;
            Ok(Node::KVValueHash(key, value, value_hash))
        }
        NodeKind::KVDigest => {
            let key = cur.read_key()?;
            let value_hash = cur.read_hash()?;
            Ok(Node::KVDigest(key, value_hash))
        }
        NodeKind::KVRefValueHash => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let value_hash = cur.read_hash()?;
            Ok(Node::KVRefValueHash(key, value, value_hash))
        }
        NodeKind::KVBackwardsReferencesValueHash => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let backrefs_hash = cur.read_hash()?;
            Ok(Node::KVBackwardsReferencesValueHash(
                key,
                value,
                backrefs_hash,
            ))
        }
        NodeKind::KVValueHashFeatureType => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let value_hash = cur.read_hash()?;
            let feature_type = decode_tree_feature_type(cur)?;
            Ok(Node::KVValueHashFeatureType(
                key,
                value,
                value_hash,
                feature_type,
            ))
        }
        NodeKind::KVCount => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let count = cur.read_u64_be()?;
            Ok(Node::KVCount(key, value, count))
        }
        NodeKind::KVHashCount => {
            let kv_hash = cur.read_hash()?;
            let count = cur.read_u64_be()?;
            Ok(Node::KVHashCount(kv_hash, count))
        }
        NodeKind::KVRefValueHashCount => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let value_hash = cur.read_hash()?;
            let count = cur.read_u64_be()?;
            Ok(Node::KVRefValueHashCount(key, value, value_hash, count))
        }
        NodeKind::KVDigestCount => {
            let key = cur.read_key()?;
            let value_hash = cur.read_hash()?;
            let count = cur.read_u64_be()?;
            Ok(Node::KVDigestCount(key, value_hash, count))
        }
        NodeKind::KVValueHashFeatureTypeWithChildHash => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let value_hash = cur.read_hash()?;
            let feature_type = decode_tree_feature_type(cur)?;
            let child_hash = cur.read_hash()?;
            Ok(Node::KVValueHashFeatureTypeWithChildHash(
                key,
                value,
                value_hash,
                feature_type,
                child_hash,
            ))
        }
        NodeKind::HashWithCount => {
            let kv_hash = cur.read_hash()?;
            let left_child_hash = cur.read_hash()?;
            let right_child_hash = cur.read_hash()?;
            let count = cur.read_u64_be()?;
            Ok(Node::HashWithCount(
                kv_hash,
                left_child_hash,
                right_child_hash,
                count,
            ))
        }
        NodeKind::KVSum => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let sum = cur.read_i64_be()?;
            Ok(Node::KVSum(key, value, sum))
        }
        NodeKind::KVHashSum => {
            let kv_hash = cur.read_hash()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::KVHashSum(kv_hash, sum))
        }
        NodeKind::KVRefValueHashSum => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let value_hash = cur.read_hash()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::KVRefValueHashSum(key, value, value_hash, sum))
        }
        NodeKind::KVDigestSum => {
            let key = cur.read_key()?;
            let value_hash = cur.read_hash()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::KVDigestSum(key, value_hash, sum))
        }
        NodeKind::HashWithSum => {
            let kv_hash = cur.read_hash()?;
            let left_child_hash = cur.read_hash()?;
            let right_child_hash = cur.read_hash()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::HashWithSum(
                kv_hash,
                left_child_hash,
                right_child_hash,
                sum,
            ))
        }
        NodeKind::KVCountSum => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let count = cur.read_u64_be()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::KVCountSum(key, value, count, sum))
        }
        NodeKind::KVHashCountSum => {
            let kv_hash = cur.read_hash()?;
            let count = cur.read_u64_be()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::KVHashCountSum(kv_hash, count, sum))
        }
        NodeKind::KVRefValueHashCountSum => {
            let key = cur.read_key()?;
            let value = cur.read_value(large, tag)?;
            let value_hash = cur.read_hash()?;
            let count = cur.read_u64_be()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::KVRefValueHashCountSum(
                key, value, value_hash, count, sum,
            ))
        }
        NodeKind::KVDigestCountSum => {
            let key = cur.read_key()?;
            let value_hash = cur.read_hash()?;
            let count = cur.read_u64_be()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::KVDigestCountSum(key, value_hash, count, sum))
        }
        NodeKind::HashWithCountAndSum => {
            let kv_hash = cur.read_hash()?;
            let left_child_hash = cur.read_hash()?;
            let right_child_hash = cur.read_hash()?;
            let count = cur.read_u64_be()?;
            let sum = cur.read_i64_be()?;
            Ok(Node::HashWithCountAndSum(
                kv_hash,
                left_child_hash,
                right_child_hash,
                count,
                sum,
            ))
        }
    }
}

/// Decode one [`Op`] from the cursor, advancing it past the bytes read.
///
/// On error the cursor position is unspecified.
pub fn decode_op_at(cur: &mut Cursor<'_>) -> Result<Op, OpDecodeError> {
    let tag = cur.read_u8()?;
    match tag {
        0x10 => Ok(Op::Parent),
        0x11 => Ok(Op::Child),
        0x12 => Ok(Op::ParentInverted),
        0x13 => Ok(Op::ChildInverted),
        _ => match classify_push_tag(tag) {
            None => Err(OpDecodeError::UnexpectedByte(tag)),
            Some(info) => {
                let node = decode_node(cur, info.kind, info.large, tag)?;
                if info.inverted {
                    Ok(Op::PushInverted(node))
                } else {
                    Ok(Op::Push(node))
                }
            }
        },
    }
}

/// Decode one [`Op`] from the start of `bytes`. Trailing bytes are ignored.
pub fn decode_op(bytes: &[u8]) -> Result<Op, OpDecodeError> {
    let mut cur = Cursor::new(bytes);
    decode_op_at(&mut cur)
}

#[cfg(test)]
mod tests {
    use ed::{Decode, Encode};
    use integer_encoding::{VarInt, VarIntReader};

    use super::*;
    use crate::error::Error;

    /// Decode with the original `ed::Decode` path, returning the result and
    /// the number of bytes it consumed.
    fn old_decode(bytes: &[u8]) -> (ed::Result<Op>, usize) {
        let mut slice = bytes;
        let res = <Op as Decode>::decode(&mut slice);
        (res, bytes.len() - slice.len())
    }

    /// The error string the original `Op::decode` produced.
    fn old_error_string(e: &ed::Error) -> String {
        match e {
            ed::Error::UnexpectedByte(byte) => {
                format!("failed to decode an proofs::Op structure (UnexpectedByte: {byte})")
            }
            ed::Error::IOError(error) => {
                format!("failed to decode an proofs::Op structure ({error})")
            }
        }
    }

    /// Assert the new and old decoders agree on `bytes`.
    fn check(bytes: &[u8]) {
        let (old, old_used) = old_decode(bytes);
        let mut cur = Cursor::new(bytes);
        let new = decode_op_at(&mut cur);
        match (&old, &new) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b, "value mismatch on {bytes:02x?}");
                assert_eq!(
                    old_used,
                    cur.position(),
                    "consumed mismatch on {bytes:02x?}"
                );
            }
            (Err(_), Err(_)) => {}
            _ => panic!("accept/reject mismatch on {bytes:02x?}: old={old:?} new={new:?}"),
        }
        // The public wrapper must produce the old error strings.
        match (old, Op::decode(bytes)) {
            (Ok(a), Ok(b)) => assert_eq!(a, b),
            (Err(e), Err(Error::InvalidProofError(s))) => {
                assert_eq!(
                    old_error_string(&e),
                    s,
                    "error string mismatch on {bytes:02x?}"
                );
            }
            (o, n) => panic!("wrapper mismatch on {bytes:02x?}: old={o:?} new={n:?}"),
        }
    }

    fn hash(seed: u8) -> [u8; HASH_LENGTH] {
        let mut h = [0u8; HASH_LENGTH];
        for (i, b) in h.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
        h
    }

    fn feature_types() -> Vec<TreeFeatureType> {
        use TreeFeatureType::*;
        vec![
            BasicMerkNode,
            SummedMerkNode(0),
            SummedMerkNode(-1),
            SummedMerkNode(i64::MIN),
            SummedMerkNode(i64::MAX),
            BigSummedMerkNode(0),
            BigSummedMerkNode(-5),
            BigSummedMerkNode(i128::MIN),
            BigSummedMerkNode(i128::MAX),
            CountedMerkNode(0),
            CountedMerkNode(300),
            CountedMerkNode(u64::MAX),
            CountedSummedMerkNode(1, -1),
            CountedSummedMerkNode(u64::MAX, i64::MIN),
            ProvableCountedMerkNode(127),
            ProvableCountedMerkNode(128),
            ProvableCountedSummedMerkNode(u64::MAX, i64::MAX),
            ProvableSummedMerkNode(-12345),
            ProvableCountedAndProvableSummedMerkNode(77, -77),
            ProvableCountedAndProvableSummedMerkNode(u64::MAX, i64::MIN),
        ]
    }

    /// One representative of every `Node` variant, for the given key/value.
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
            Node::KVCount(k.clone(), v.clone(), 0x0102030405060708),
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
        for ft in feature_types() {
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

    fn all_ops(key: &[u8], value: &[u8]) -> Vec<Op> {
        let mut ops = vec![Op::Parent, Op::Child, Op::ParentInverted, Op::ChildInverted];
        for node in all_nodes(key, value) {
            ops.push(Op::Push(node.clone()));
            ops.push(Op::PushInverted(node));
        }
        ops
    }

    fn encode(op: &Op) -> Vec<u8> {
        let mut out = vec![];
        op.encode_into(&mut out).unwrap();
        out
    }

    /// Small ops (short values) plus one set with a >= 64KiB value so the
    /// large (u32 length) tags are produced by the encoder.
    fn corpus() -> Vec<Vec<u8>> {
        let mut out = vec![];
        for op in all_ops(&[1, 2, 3], &[4, 5, 6]) {
            out.push(encode(&op));
        }
        for op in all_ops(&[], &[]) {
            out.push(encode(&op));
        }
        let long_key = vec![0xabu8; 255];
        for op in all_ops(&long_key, &[7u8; 300]) {
            out.push(encode(&op));
        }
        // Large (u32 length) variants.
        let big = vec![0x5au8; 65536];
        for op in all_ops(&[9, 9], &big) {
            out.push(encode(&op));
        }
        out
    }

    #[test]
    fn round_trip_every_variant() {
        for op in all_ops(&[1, 2, 3], &[4, 5, 6])
            .into_iter()
            .chain(all_ops(&[9, 9], &vec![1u8; 65536]))
        {
            let bytes = encode(&op);
            assert_eq!(decode_op(&bytes), Ok(op.clone()));
            let mut cur = Cursor::new(&bytes);
            decode_op_at(&mut cur).unwrap();
            assert_eq!(cur.position(), bytes.len());
        }
    }

    #[test]
    fn corpus_contains_every_tag() {
        // Every tag the old decoder accepts must be exercised by the corpus.
        let mut seen = [false; 256];
        for bytes in corpus() {
            seen[bytes[0] as usize] = true;
        }
        for tag in 0..=255u8 {
            let mut probe = vec![tag];
            probe.extend_from_slice(&[0u8; 400]);
            let (old, _) = old_decode(&probe);
            // A zero-filled tail decodes for every valid tag except those
            // whose first byte combos are rejected; treat "accepted by old
            // decoder with zero filler" as the valid set.
            if old.is_ok() {
                assert!(seen[tag as usize], "tag {tag:#04x} not in corpus");
            }
        }
    }

    #[test]
    fn corpus_agrees() {
        for bytes in corpus() {
            check(&bytes);
        }
    }

    #[test]
    fn corpus_with_trailing_bytes_agrees() {
        for mut bytes in corpus() {
            bytes.extend_from_slice(&[0x10, 0xff, 0x00, 0x51]);
            check(&bytes);
        }
    }

    #[test]
    fn every_first_byte_agrees() {
        // All 256 tags against several fillers, including empty / short.
        let fillers: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            vec![0; 3],
            vec![0; 400],
            vec![0xff; 400],
            vec![0x01; 400],
            vec![0x80; 400],
        ];
        for tag in 0..=255u8 {
            for filler in &fillers {
                let mut bytes = vec![tag];
                bytes.extend_from_slice(filler);
                check(&bytes);
            }
        }
        check(&[]);
    }

    #[test]
    fn all_truncations_agree() {
        for bytes in corpus() {
            // Skip the very long ones past a stride to keep the test fast but
            // always cover the first/last 600 bytes of every encoding.
            for cut in 0..bytes.len() {
                if cut > 600 && cut + 600 < bytes.len() && cut % 4099 != 0 {
                    continue;
                }
                check(&bytes[..cut]);
            }
        }
    }

    #[test]
    fn single_byte_mutations_agree() {
        for bytes in corpus() {
            if bytes.len() > 2000 {
                // Large encodings: mutate the header and the tail only.
                for pos in (0..16).chain(bytes.len() - 80..bytes.len()) {
                    for delta in [0x01u8, 0x80, 0xff] {
                        let mut m = bytes.clone();
                        m[pos] ^= delta;
                        check(&m);
                    }
                }
                continue;
            }
            for pos in 0..bytes.len() {
                for replacement in [0x00u8, 0x01, 0x02, 0x7f, 0x80, 0x81, 0xff] {
                    let mut m = bytes.clone();
                    m[pos] = replacement;
                    check(&m);
                }
            }
        }
    }

    #[test]
    fn non_canonical_lengths_agree() {
        // Large tags carrying lengths that would fit a u16, and small/large
        // length boundaries.
        for tag in [0x20u8, 0x51, 0x53, 0x31] {
            for len in [0u32, 1, 3, 255, 256, 65535, 65536, 65537] {
                for key_len in [0u8, 2] {
                    let mut bytes = vec![tag, key_len];
                    bytes.extend(std::iter::repeat_n(7u8, key_len as usize));
                    bytes.extend_from_slice(&len.to_be_bytes());
                    bytes.extend(std::iter::repeat_n(1u8, len as usize));
                    bytes.extend_from_slice(&[0x33u8; 40]);
                    check(&bytes);
                }
            }
        }
        // Explicit: tag 0x51 with u32 length 3 and a 32-byte hash.
        let mut bytes = vec![0x51, 1, 0xaa, 0, 0, 0, 3, 1, 2, 3];
        bytes.extend_from_slice(&[9u8; 32]);
        check(&bytes);
        assert!(matches!(
            decode_op(&bytes),
            Ok(Op::Push(Node::KVBackwardsReferencesValueHash(_, _, _)))
        ));
        // Over the maximum, for every large tag, with and without data.
        let large_tags = [
            0x20u8, 0x21, 0x22, 0x23, 0x24, 0x25, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
            0x31, 0x34, 0x38, 0x3b, 0x41, 0x44, 0x48, 0x4b, 0x51, 0x53,
        ];
        for tag in large_tags {
            for len in [MAX_VALUE_LEN, MAX_VALUE_LEN + 1, u32::MAX] {
                let mut bytes = vec![tag, 0];
                bytes.extend_from_slice(&len.to_be_bytes());
                bytes.extend_from_slice(&[0u8; 64]);
                check(&bytes);
                if len > MAX_VALUE_LEN {
                    assert_eq!(decode_op(&bytes), Err(OpDecodeError::UnexpectedByte(tag)));
                }
            }
        }
        // Non-canonical varints inside feature types: padded zero bytes.
        for ft_tag in 0u8..=9 {
            for payload in [
                vec![0x80, 0x00],
                vec![0x80, 0x80, 0x00],
                vec![0xff, 0x00],
                vec![0x00],
                vec![0x01, 0x80, 0x00],
            ] {
                let mut bytes = vec![0x07, 0];
                bytes.extend_from_slice(&0u16.to_be_bytes());
                bytes.extend_from_slice(&[5u8; 32]);
                bytes.push(ft_tag);
                bytes.extend_from_slice(&payload);
                bytes.extend_from_slice(&[0u8; 20]);
                check(&bytes);
            }
        }
    }

    /// Compare `read_varint_*` against `integer-encoding` on raw bytes.
    fn check_varint(bytes: &[u8]) {
        let mut slice = bytes;
        let old_u: std::io::Result<u64> = slice.read_varint();
        let used_u = bytes.len() - slice.len();
        let mut cur = Cursor::new(bytes);
        let new_u = cur.read_varint_u64();
        match (&old_u, &new_u) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b, "u64 varint value on {bytes:02x?}");
                assert_eq!(
                    used_u,
                    cur.position(),
                    "u64 varint consumed on {bytes:02x?}"
                );
            }
            (Err(e), Err(n)) => {
                let expected = match n {
                    OpDecodeError::TruncatedVarint => "Reached EOF",
                    OpDecodeError::UnterminatedVarint => "Unterminated varint",
                    _ => panic!("unexpected error kind {n:?}"),
                };
                assert_eq!(e.to_string(), expected, "u64 varint error on {bytes:02x?}");
            }
            _ => panic!("u64 varint mismatch on {bytes:02x?}: {old_u:?} vs {new_u:?}"),
        }

        let mut slice = bytes;
        let old_i: std::io::Result<i64> = slice.read_varint();
        let used_i = bytes.len() - slice.len();
        let mut cur = Cursor::new(bytes);
        let new_i = cur.read_varint_i64();
        match (&old_i, &new_i) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b, "i64 varint value on {bytes:02x?}");
                assert_eq!(
                    used_i,
                    cur.position(),
                    "i64 varint consumed on {bytes:02x?}"
                );
            }
            (Err(e), Err(n)) => {
                let expected = match n {
                    OpDecodeError::TruncatedVarint => "Reached EOF",
                    OpDecodeError::UnterminatedVarint => "Unterminated varint",
                    _ => panic!("unexpected error kind {n:?}"),
                };
                assert_eq!(e.to_string(), expected, "i64 varint error on {bytes:02x?}");
            }
            _ => panic!("i64 varint mismatch on {bytes:02x?}: {old_i:?} vs {new_i:?}"),
        }
    }

    #[test]
    fn varint_edge_cases() {
        let mut cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![0x00],
            vec![0x7f],
            vec![0x80],
            vec![0x80, 0x00],
            vec![0x80, 0x80, 0x80],
            vec![0xff; 9],
        ];
        // 10-byte varints with last byte 0, 1, 2, 3, 0x7f, 0x80, 0xff.
        for last in [0x00u8, 0x01, 0x02, 0x03, 0x7f, 0x80, 0x81, 0xff] {
            let mut v = vec![0xffu8; 9];
            v.push(last);
            cases.push(v.clone());
            // followed by more bytes (unterminated when last has MSB set)
            let mut longer = v.clone();
            longer.push(0x00);
            cases.push(longer);
            let mut longer2 = v;
            longer2.extend_from_slice(&[0x80, 0x01]);
            cases.push(longer2);
        }
        // Overlong: zero payload padded with continuation bytes.
        for n in 1..=12usize {
            let mut v = vec![0x80u8; n];
            v.push(0x00);
            cases.push(v);
            cases.push(vec![0x80u8; n]);
            cases.push(vec![0xffu8; n]);
        }
        // Canonical encodings of interesting values.
        for v in [
            0u64,
            1,
            127,
            128,
            255,
            16383,
            16384,
            u32::MAX as u64,
            u64::MAX - 1,
            u64::MAX,
            1 << 63,
        ] {
            cases.push(v.encode_var_vec());
        }
        for v in [
            0i64,
            1,
            -1,
            63,
            -64,
            64,
            -65,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
        ] {
            cases.push(v.encode_var_vec());
        }
        for case in cases {
            check_varint(&case);
        }
    }

    #[test]
    fn varint_exhaustive_short() {
        // All 1- and 2-byte inputs.
        for a in 0..=255u8 {
            check_varint(&[a]);
            for b in 0..=255u8 {
                check_varint(&[a, b]);
            }
        }
    }

    /// splitmix64: tiny fixed-seed PRNG so the tests need no extra deps.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn varint_random_agrees() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for _ in 0..200_000 {
            let len = (rng.next() % 13) as usize;
            let mut bytes = Vec::with_capacity(len);
            for _ in 0..len {
                let r = rng.next();
                // Bias toward continuation bytes so long varints occur.
                let b = if r & 0x300 == 0 {
                    (r >> 16) as u8 & 0x7f
                } else {
                    (r >> 16) as u8 | 0x80
                };
                bytes.push(b);
            }
            check_varint(&bytes);
        }
    }

    #[test]
    fn random_byte_strings_agree() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        // Fully random bytes (mostly rejected by tag/EOF).
        for _ in 0..4_000 {
            let len = (rng.next() % 120) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            check(&bytes);
        }
        // Valid tag with short random bodies; keep lengths small so most
        // inputs get past the headers.
        let valid_tags: Vec<u8> = (0..=255u8)
            .filter(|t| {
                let mut p = vec![*t];
                p.extend_from_slice(&[0u8; 400]);
                old_decode(&p).0.is_ok()
            })
            .collect();
        assert!(valid_tags.len() > 60);
        for _ in 0..8_000 {
            let tag = valid_tags[(rng.next() as usize) % valid_tags.len()];
            let body_len = (rng.next() % 150) as usize;
            let mut bytes = vec![tag];
            for _ in 0..body_len {
                let r = rng.next();
                // Mostly small bytes so lengths are small, sometimes anything.
                let b = if r & 0x3 == 0 {
                    (r >> 8) as u8
                } else {
                    (r >> 8) as u8 & 0x0f
                };
                bytes.push(b);
            }
            check(&bytes);
        }
        // Mutations of valid encodings with random byte replacements.
        let corpus: Vec<Vec<u8>> = corpus().into_iter().filter(|b| b.len() < 2000).collect();
        for _ in 0..8_000 {
            let mut bytes = corpus[(rng.next() as usize) % corpus.len()].clone();
            let flips = 1 + rng.next() % 3;
            for _ in 0..flips {
                let pos = (rng.next() as usize) % bytes.len();
                bytes[pos] = rng.next() as u8;
            }
            check(&bytes);
        }
    }

    #[test]
    fn decoder_advances_by_encoding_length_not_bytes_consumed() {
        use crate::proofs::Decoder;

        // Non-canonical u32 length (3 fits a u16): decoding consumes 10
        // bytes, but `Op::encoding_length` is 8, so the decoder resumes two
        // bytes early, inside the value.
        let bytes = vec![0x20, 1, 0xaa, 0, 0, 0, 3, 1, 2, 3, 0x10];
        let mut decoder = Decoder::new(&bytes);
        assert_eq!(
            decoder.next().unwrap().unwrap(),
            Op::Push(Node::KV(vec![0xaa], vec![1, 2, 3]))
        );
        assert_eq!(decoder.remaining_bytes(), 3);
        assert!(decoder.next().unwrap().is_err());
        assert!(decoder.next().is_none());

        // Padded varint inside a feature type: 2 bytes consumed, canonical
        // length is 1, so the decoder resumes on the padding terminator.
        let mut bytes = vec![0x07, 0, 0, 0];
        bytes.extend_from_slice(&[5u8; 32]);
        bytes.extend_from_slice(&[3, 0x80, 0x00, 0x10]);
        let mut decoder = Decoder::new(&bytes);
        assert_eq!(
            decoder.next().unwrap().unwrap(),
            Op::Push(Node::KVValueHashFeatureType(
                vec![],
                vec![],
                [5u8; 32],
                TreeFeatureType::CountedMerkNode(0)
            ))
        );
        assert_eq!(decoder.remaining_bytes(), 2);
        assert!(matches!(
            decoder.next(),
            Some(Err(Error::InvalidProofError(_)))
        ));
        assert!(decoder.next().is_none());
    }

    /// The decoder loop as it was before `next_core`: `Op::decode` on the
    /// remaining bytes, advancing by `ed`'s `encoding_length`.
    fn reference_decoder(bytes: &[u8]) -> Vec<(Result<Op, String>, usize)> {
        let mut out = vec![];
        let mut offset = 0usize;
        let mut failed = false;
        while !failed && offset < bytes.len() {
            match Op::decode(&bytes[offset..]) {
                Ok(op) => {
                    offset += <Op as Encode>::encoding_length(&op).unwrap();
                    out.push((Ok(op), bytes.len().saturating_sub(offset)));
                }
                Err(e) => {
                    failed = true;
                    out.push((Err(e.to_string()), bytes.len().saturating_sub(offset)));
                }
            }
        }
        out
    }

    fn check_decoder(bytes: &[u8]) {
        use crate::proofs::Decoder;
        let expected = reference_decoder(bytes);
        let mut decoder = Decoder::new(bytes);
        let mut core_decoder = Decoder::new(bytes);
        for (item, remaining) in &expected {
            let got = decoder.next().expect("item expected");
            let got_core = core_decoder.next_core().expect("core item expected");
            match (item, &got, &got_core) {
                (Ok(a), Ok(b), Ok(c)) => {
                    assert_eq!(a, b);
                    assert_eq!(a, c);
                }
                (Err(a), Err(b), Err(c)) => {
                    assert_eq!(a, &b.to_string());
                    assert_eq!(
                        a,
                        &format!(
                            "invalid proof error {}",
                            crate::proofs::op_decode_error_message(c)
                        )
                    );
                }
                _ => panic!("decoder mismatch on {bytes:02x?}"),
            }
            assert_eq!(*remaining, decoder.remaining_bytes(), "on {bytes:02x?}");
            assert_eq!(*remaining, core_decoder.remaining_bytes());
        }
        // Fused: exhausted after the last item (end of input or one error).
        for _ in 0..3 {
            assert!(decoder.next().is_none());
            assert!(core_decoder.next_core().is_none());
        }
    }

    #[test]
    fn decoder_matches_reference_loop() {
        let corpus: Vec<Vec<u8>> = corpus().into_iter().filter(|b| b.len() < 2000).collect();
        // Valid streams: consecutive encodings.
        let mut stream = vec![];
        for chunk in &corpus {
            stream.extend_from_slice(chunk);
        }
        check_decoder(&stream);
        // A stream with a 6-byte-header (u32 length) op in the middle.
        let big = encode(&Op::Push(Node::KV(vec![1], vec![9u8; 65536])));
        let mut with_big = encode(&Op::Parent);
        with_big.extend_from_slice(&big);
        with_big.extend_from_slice(&encode(&Op::Child));
        check_decoder(&with_big);
        check_decoder(&with_big[..with_big.len() - 1]);
        for end in [0usize, 1, 7, 100, stream.len() / 2, stream.len() - 1] {
            check_decoder(&stream[..end]);
        }
        // Individual encodings with trailing/mutated bytes, plus
        // non-canonical ones (offset undershoot) followed by more ops.
        let mut rng = Rng(0x0dec_0de5);
        for _ in 0..3_000 {
            let mut bytes = vec![];
            let parts = 1 + rng.next() % 4;
            for _ in 0..parts {
                bytes.extend_from_slice(&corpus[(rng.next() as usize) % corpus.len()]);
            }
            let flips = rng.next() % 3;
            for _ in 0..flips {
                let pos = (rng.next() as usize) % bytes.len();
                bytes[pos] = rng.next() as u8;
            }
            if rng.next().is_multiple_of(3) {
                let cut = (rng.next() as usize) % bytes.len();
                bytes.truncate(cut);
            }
            check_decoder(&bytes);
        }
        for _ in 0..3_000 {
            let len = (rng.next() % 120) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            check_decoder(&bytes);
        }
        // Non-canonical encodings followed by more ops.
        check_decoder(&[0x20, 1, 0xaa, 0, 0, 0, 3, 1, 2, 3, 0x10, 0x10, 0x11]);
        check_decoder(&[0x51, 0, 0, 0, 0, 0]);
        let mut padded = vec![0x07, 0, 0, 0];
        padded.extend_from_slice(&[5u8; 32]);
        padded.extend_from_slice(&[3, 0x80, 0x00, 0x10, 0x11, 0x10]);
        check_decoder(&padded);
    }

    #[test]
    fn wrapper_error_strings_are_unchanged() {
        let cases: Vec<(Vec<u8>, &str)> = vec![
            (
                vec![],
                "failed to decode an proofs::Op structure (failed to fill whole buffer)",
            ),
            (
                vec![0xff],
                "failed to decode an proofs::Op structure (UnexpectedByte: 255)",
            ),
            (
                vec![0x01, 1, 2, 3],
                "failed to decode an proofs::Op structure (failed to fill whole buffer)",
            ),
            (
                vec![0x20, 0, 0xff, 0xff, 0xff, 0xff],
                "failed to decode an proofs::Op structure (UnexpectedByte: 32)",
            ),
            (
                vec![0x07, 0, 0, 0]
                    .into_iter()
                    .chain([0u8; 32])
                    .chain([9])
                    .collect(),
                "failed to decode an proofs::Op structure (UnexpectedByte: 9)",
            ),
            (
                vec![0x07, 0, 0, 0]
                    .into_iter()
                    .chain([0u8; 32])
                    .chain([1])
                    .collect(),
                "failed to decode an proofs::Op structure (Reached EOF)",
            ),
            (
                vec![0x07, 0, 0, 0]
                    .into_iter()
                    .chain([0u8; 32])
                    .chain([3])
                    .chain([0xff; 11])
                    .collect(),
                "failed to decode an proofs::Op structure (Unterminated varint)",
            ),
            (
                vec![0x07, 0, 0, 0]
                    .into_iter()
                    .chain([0u8; 32])
                    .chain([3])
                    .chain([0xff; 9])
                    .chain([0x02])
                    .collect(),
                "failed to decode an proofs::Op structure (Reached EOF)",
            ),
        ];
        for (bytes, expected) in cases {
            match Op::decode(&bytes) {
                Err(Error::InvalidProofError(s)) => assert_eq!(s, expected),
                other => panic!("expected InvalidProofError on {bytes:02x?}, got {other:?}"),
            }
            check(&bytes);
        }
    }

    #[test]
    fn integer_reads_match_ed() {
        let mut rng = Rng(42);
        for _ in 0..2_000 {
            let len = (rng.next() % 20) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            macro_rules! cmp {
                ($ty:ty, $meth:ident) => {{
                    let mut s = &bytes[..];
                    let old = <$ty as Decode>::decode(&mut s);
                    let mut cur = Cursor::new(&bytes);
                    let new = cur.$meth();
                    match (old, new) {
                        (Ok(a), Ok(b)) => {
                            assert_eq!(a, b);
                            assert_eq!(bytes.len() - s.len(), cur.position());
                        }
                        (Err(_), Err(OpDecodeError::UnexpectedEof)) => {}
                        (o, n) => panic!("mismatch {o:?} {n:?}"),
                    }
                }};
            }
            cmp!(u16, read_u16_be);
            cmp!(u32, read_u32_be);
            cmp!(u64, read_u64_be);
            cmp!(i64, read_i64_be);
            cmp!(i128, read_i128_be);
        }
    }
}
