use crate::{Element, Error, GroveDb};

impl GroveDb {
    pub fn delete(&mut self, path: &[&[u8]], key: Vec<u8>) -> Result<(), Error> {
        let element = self.get_raw(path, &key)?;
        if path.is_empty() {
            // Attempt to delete a root tree leaf
            Err(Error::InvalidPath(
                "root tree leafs currently cannot be deleted",
            ))
        } else {
            let mut merk = self
                .subtrees
                .get_mut(&Self::compress_subtree_key(path, None))
                .ok_or(Error::InvalidPath("no subtree found under that path"))?;
            Element::delete(&mut merk, key.clone())?;
            if let Element::Tree(_) = element {
                // TODO: dumb traversal should not be tolerated
                let mut concat_path: Vec<Vec<u8>> = path.iter().map(|x| x.to_vec()).collect();
                concat_path.push(key);
                let subtrees_paths = self.find_subtrees(concat_path)?;
                for subtree_path in subtrees_paths {
                    // TODO: eventually we need to do something about this nested slices
                    let subtree_path_ref: Vec<&[u8]> =
                        subtree_path.iter().map(|x| x.as_slice()).collect();
                    let prefix = Self::compress_subtree_key(&subtree_path_ref, None);
                    if let Some(subtree) = self.subtrees.remove(&prefix) {
                        subtree.clear().map_err(|e| {
                            Error::CorruptedData(format!(
                                "unable to cleanup tree from storage: {}",
                                e
                            ))
                        })?;
                    }
                }
            }
            self.propagate_changes(path)?;
            Ok(())
        }
    }

    // TODO: dumb traversal should not be tolerated
    /// Finds keys which are trees for a given subtree recursively.
    /// One element means a key of a `merk`, n > 1 elements mean relative path
    /// for a deeply nested subtree.
    pub(crate) fn find_subtrees(&self, path: Vec<Vec<u8>>) -> Result<Vec<Vec<Vec<u8>>>, Error> {
        let mut queue: Vec<Vec<Vec<u8>>> = vec![path.clone()];
        let mut result: Vec<Vec<Vec<u8>>> = vec![path.clone()];

        while let Some(q) = queue.pop() {
            // TODO: eventually we need to do something about this nested slices
            let q_ref: Vec<&[u8]> = q.iter().map(|x| x.as_slice()).collect();
            let mut iter = self.elements_iterator(&q_ref)?;
            while let Some((key, value)) = iter.next()? {
                match value {
                    Element::Tree(_) => {
                        let mut sub_path = q.clone();
                        sub_path.push(key);
                        queue.push(sub_path.clone());
                        result.push(sub_path);
                    }
                    _ => {}
                }
            }
        }
        Ok(result)
    }
}