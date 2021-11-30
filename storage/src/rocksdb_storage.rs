//! Storage implementation using RocksDB
use std::rc::Rc;

use rocksdb::{ColumnFamily, ColumnFamilyDescriptor, DBRawIterator, WriteBatch};
pub use rocksdb::{Error, DB};

use crate::{Batch, RawIterator, Storage};

const AUX_CF_NAME: &str = "aux";
const ROOTS_CF_NAME: &str = "roots";
const META_CF_NAME: &str = "meta";

/// RocksDB options
pub fn default_db_opts() -> rocksdb::Options {
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(true);
    opts.increase_parallelism(num_cpus::get() as i32);
    opts.set_allow_mmap_writes(true);
    opts.set_allow_mmap_reads(true);
    opts.create_missing_column_families(true);
    opts.set_atomic_flush(true);
    opts
}

/// RocksDB column families
pub fn column_families() -> Vec<ColumnFamilyDescriptor> {
    vec![
        ColumnFamilyDescriptor::new(AUX_CF_NAME, default_db_opts()),
        ColumnFamilyDescriptor::new(ROOTS_CF_NAME, default_db_opts()),
        ColumnFamilyDescriptor::new(META_CF_NAME, default_db_opts()),
    ]
}

fn make_prefixed_key(prefix: Vec<u8>, key: &[u8]) -> Vec<u8> {
    let mut prefixed_key = prefix.clone();
    prefixed_key.extend_from_slice(key);
    prefixed_key
}

/// RocksDB wrapper to store items with prefixes
pub struct PrefixedRocksDbStorage {
    db: Rc<rocksdb::DB>,
    prefix: Vec<u8>,
}

#[derive(thiserror::Error, Debug)]
pub enum PrefixedRocksDbStorageError {
    #[error("column family not found: {0}")]
    ColumnFamilyNotFound(&'static str),
    #[error(transparent)]
    RocksDbError(#[from] rocksdb::Error),
}

impl PrefixedRocksDbStorage {
    /// Wraps RocksDB to prepend prefixes to each operation
    pub fn new(db: Rc<rocksdb::DB>, prefix: Vec<u8>) -> Result<Self, PrefixedRocksDbStorageError> {
        Ok(PrefixedRocksDbStorage { prefix, db })
    }

    /// Get auxiliary data column family
    fn cf_aux(&self) -> Result<&rocksdb::ColumnFamily, PrefixedRocksDbStorageError> {
        self.db
            .cf_handle(AUX_CF_NAME)
            .ok_or(PrefixedRocksDbStorageError::ColumnFamilyNotFound(
                AUX_CF_NAME,
            ))
    }

    /// Get trees roots data column family
    fn cf_roots(&self) -> Result<&rocksdb::ColumnFamily, PrefixedRocksDbStorageError> {
        self.db
            .cf_handle(ROOTS_CF_NAME)
            .ok_or(PrefixedRocksDbStorageError::ColumnFamilyNotFound(
                ROOTS_CF_NAME,
            ))
    }

    /// Get metadata column family
    fn cf_meta(&self) -> Result<&rocksdb::ColumnFamily, PrefixedRocksDbStorageError> {
        self.db
            .cf_handle(META_CF_NAME)
            .ok_or(PrefixedRocksDbStorageError::ColumnFamilyNotFound(
                META_CF_NAME,
            ))
    }
}

impl Storage for PrefixedRocksDbStorage {
    type Batch<'a> = PrefixedRocksDbBatch<'a>;
    type Error = PrefixedRocksDbStorageError;
    type RawIterator<'a> = rocksdb::DBRawIterator<'a>;

    fn put(&self, key: &[u8], value: &[u8]) -> Result<(), Self::Error> {
        self.db
            .put(make_prefixed_key(self.prefix.clone(), key), value)?;
        Ok(())
    }

    fn put_aux(&self, key: &[u8], value: &[u8]) -> Result<(), Self::Error> {
        self.db.put_cf(
            self.cf_aux()?,
            make_prefixed_key(self.prefix.clone(), key),
            value,
        )?;
        Ok(())
    }

    fn put_root(&self, key: &[u8], value: &[u8]) -> Result<(), Self::Error> {
        self.db.put_cf(
            self.cf_roots()?,
            make_prefixed_key(self.prefix.clone(), key),
            value,
        )?;
        Ok(())
    }

    fn delete(&self, key: &[u8]) -> Result<(), Self::Error> {
        self.db
            .delete(make_prefixed_key(self.prefix.clone(), key))?;
        Ok(())
    }

    fn delete_aux(&self, key: &[u8]) -> Result<(), Self::Error> {
        self.db
            .delete_cf(self.cf_aux()?, make_prefixed_key(self.prefix.clone(), key))?;
        Ok(())
    }

    fn delete_root(&self, key: &[u8]) -> Result<(), Self::Error> {
        self.db.delete_cf(
            self.cf_roots()?,
            make_prefixed_key(self.prefix.clone(), key),
        )?;
        Ok(())
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.db.get(make_prefixed_key(self.prefix.clone(), key))?)
    }

    fn get_aux(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self
            .db
            .get_cf(self.cf_aux()?, make_prefixed_key(self.prefix.clone(), key))?)
    }

    fn get_root(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.db.get_cf(
            self.cf_roots()?,
            make_prefixed_key(self.prefix.clone(), key),
        )?)
    }

    fn put_meta(&self, key: &[u8], value: &[u8]) -> Result<(), Self::Error> {
        Ok(self.db.put_cf(self.cf_meta()?, key, value)?)
    }

    fn delete_meta(&self, key: &[u8]) -> Result<(), Self::Error> {
        Ok(self.db.delete_cf(self.cf_meta()?, key)?)
    }

    fn get_meta(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.db.get_cf(self.cf_meta()?, key)?)
    }

    fn new_batch<'a>(&'a self) -> Result<Self::Batch<'a>, Self::Error> {
        Ok(PrefixedRocksDbBatch {
            prefix: self.prefix.clone(),
            batch: WriteBatch::default(),
            cf_aux: self.cf_aux()?,
            cf_roots: self.cf_roots()?,
        })
    }

    fn commit_batch<'a>(&'a self, batch: Self::Batch<'a>) -> Result<(), Self::Error> {
        self.db.write(batch.batch)?;
        Ok(())
    }

    fn flush(&self) -> Result<(), Self::Error> {
        self.db.flush()?;
        Ok(())
    }

    fn raw_iter<'a>(&'a self) -> Self::RawIterator<'a> {
        self.db.raw_iterator()
    }
}

impl RawIterator for rocksdb::DBRawIterator<'_> {
    fn seek_to_first(&mut self) {
        DBRawIterator::seek_to_first(self)
    }

    fn seek(&mut self, key: &[u8]) {
        DBRawIterator::seek(self, key)
    }

    fn next(&mut self) {
        DBRawIterator::next(self)
    }

    fn value(&self) -> Option<&[u8]> {
        DBRawIterator::value(self)
    }

    fn key(&self) -> Option<&[u8]> {
        DBRawIterator::key(self)
    }

    fn valid(&self) -> bool {
        DBRawIterator::valid(self)
    }
}

/// Wrapper to RocksDB batch
pub struct PrefixedRocksDbBatch<'a> {
    prefix: Vec<u8>,
    batch: rocksdb::WriteBatch,
    cf_aux: &'a ColumnFamily,
    cf_roots: &'a ColumnFamily,
}

impl<'a> Batch for PrefixedRocksDbBatch<'a> {
    fn put(&mut self, key: &[u8], value: &[u8]) {
        self.batch
            .put(make_prefixed_key(self.prefix.clone(), key), value)
    }

    fn put_aux(&mut self, key: &[u8], value: &[u8]) {
        self.batch.put_cf(
            self.cf_aux,
            make_prefixed_key(self.prefix.clone(), key),
            value,
        )
    }

    fn put_root(&mut self, key: &[u8], value: &[u8]) {
        self.batch.put_cf(
            self.cf_roots,
            make_prefixed_key(self.prefix.clone(), key),
            value,
        )
    }

    fn delete(&mut self, key: &[u8]) {
        self.batch
            .delete(make_prefixed_key(self.prefix.clone(), key))
    }

    fn delete_aux(&mut self, key: &[u8]) {
        self.batch
            .delete_cf(self.cf_aux, make_prefixed_key(self.prefix.clone(), key))
    }

    fn delete_root(&mut self, key: &[u8]) {
        self.batch
            .delete_cf(self.cf_roots, make_prefixed_key(self.prefix.clone(), key))
    }
}