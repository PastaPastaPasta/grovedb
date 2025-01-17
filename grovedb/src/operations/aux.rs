use storage::StorageContext;

use crate::{util::meta_storage_context_optional_tx, Error, GroveDb, TransactionArg};

impl GroveDb {
    pub fn put_aux<K: AsRef<[u8]>>(
        &self,
        key: K,
        value: &[u8],
        transaction: TransactionArg,
    ) -> Result<(), Error> {
        meta_storage_context_optional_tx!(self.db, transaction, aux_storage, {
            aux_storage.put_aux(key, value)?;
        });
        Ok(())
    }

    pub fn delete_aux<K: AsRef<[u8]>>(
        &self,
        key: K,
        transaction: TransactionArg,
    ) -> Result<(), Error> {
        meta_storage_context_optional_tx!(self.db, transaction, aux_storage, {
            aux_storage.delete_aux(key)?;
        });
        Ok(())
    }

    pub fn get_aux<K: AsRef<[u8]>>(
        &self,
        key: K,
        transaction: TransactionArg,
    ) -> Result<Option<Vec<u8>>, Error> {
        meta_storage_context_optional_tx!(self.db, transaction, aux_storage, {
            Ok(aux_storage.get_aux(key)?)
        })
    }
}
