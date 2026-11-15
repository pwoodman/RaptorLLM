//! Inverted index over retrieval chunks: term -> the chunks that contain it.
//!
//! BM25 only needs the chunks that share a term with the query, so lexical search reads
//! these lists instead of scanning every chunk. The index is maintained incrementally:
//! a file that is re-parsed removes its old postings and adds the new ones.

use crate::index::Chunk;
use std::collections::{BTreeMap, HashSet};

/// Where a chunk lives: its file, its position in that file's chunk list, and its length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub path: String,
    pub idx: usize,
    /// Sum of term frequencies (BM25 document length).
    pub len: u32,
}

/// Term postings plus the chunk table they point into.
#[derive(Debug, Default)]
pub struct Postings {
    /// term -> (chunk slot, term frequency). Order within a list is not meaningful.
    terms: BTreeMap<String, Vec<(u32, u32)>>,
    /// Chunk slots. Freed slots are reused, so the table stays as large as the live chunks.
    slots: Vec<Option<Slot>>,
    free: Vec<u32>,
    /// File path -> the slots its chunks occupy, in chunk order.
    by_file: BTreeMap<String, Vec<u32>>,
    live: usize,
    total_len: u64,
}

impl Postings {
    /// Adds the chunks of `path`. Call [`Postings::remove_file`] first when replacing a file.
    pub fn add_file(&mut self, path: &str, chunks: &[Chunk]) {
        let mut slots = Vec::with_capacity(chunks.len());
        for (idx, chunk) in chunks.iter().enumerate() {
            let slot = match self.free.pop() {
                Some(s) => s,
                None => {
                    self.slots.push(None);
                    (self.slots.len() - 1) as u32
                }
            };
            self.slots[slot as usize] = Some(Slot {
                path: path.to_string(),
                idx,
                len: chunk.len,
            });
            for (term, tf) in &chunk.terms {
                self.terms
                    .entry(term.clone())
                    .or_default()
                    .push((slot, *tf));
            }
            self.live += 1;
            self.total_len += u64::from(chunk.len);
            slots.push(slot);
        }
        self.by_file.insert(path.to_string(), slots);
    }

    /// Removes the postings of `path`. `chunks` must be the chunks that were added for it.
    pub fn remove_file(&mut self, path: &str, chunks: &[Chunk]) {
        let Some(slots) = self.by_file.remove(path) else {
            return;
        };
        let dead: HashSet<u32> = slots.iter().copied().collect();
        let mut seen_terms: BTreeMap<&str, ()> = BTreeMap::new();
        for chunk in chunks {
            for term in chunk.terms.keys() {
                seen_terms.insert(term.as_str(), ());
            }
        }
        for term in seen_terms.keys() {
            if let Some(list) = self.terms.get_mut(*term) {
                list.retain(|(slot, _)| !dead.contains(slot));
                if list.is_empty() {
                    self.terms.remove(*term);
                }
            }
        }
        for slot in slots {
            if let Some(old) = self.slots[slot as usize].take() {
                self.live -= 1;
                self.total_len -= u64::from(old.len);
            }
            self.free.push(slot);
        }
    }

    /// Chunk postings for `term`: each entry is (slot, term frequency).
    pub fn postings(&self, term: &str) -> Option<&[(u32, u32)]> {
        self.terms.get(term).map(Vec::as_slice)
    }

    /// The chunk behind a slot, if it is live.
    pub fn slot(&self, slot: u32) -> Option<&Slot> {
        self.slots.get(slot as usize).and_then(Option::as_ref)
    }

    /// Number of live chunks (BM25 `N`).
    pub fn live(&self) -> usize {
        self.live
    }

    /// Mean chunk length (BM25 `avgdl`), 0 when there are no chunks.
    pub fn avg_len(&self) -> f64 {
        if self.live == 0 {
            0.0
        } else {
            self.total_len as f64 / self.live as f64
        }
    }

    /// Number of distinct terms with at least one live posting.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::build_chunks;
    use crate::parse::FileSymbols;
    use crate::retrieve::NoEmbedder;

    fn chunks(path: &str, text: &str) -> Vec<Chunk> {
        build_chunks(path, text, &FileSymbols::default(), &NoEmbedder).expect("chunks")
    }

    #[test]
    fn postings_list_every_chunk_containing_a_term() {
        let mut p = Postings::default();
        let a = chunks("a.rs", "refund refund ledger\n");
        let b = chunks("b.rs", "ledger only\n");
        p.add_file("a.rs", &a);
        p.add_file("b.rs", &b);
        let refund = p.postings("refund").expect("refund");
        assert_eq!(refund.len(), 1);
        assert_eq!(p.slot(refund[0].0).map(|s| s.path.as_str()), Some("a.rs"));
        assert_eq!(p.postings("ledger").map(<[_]>::len), Some(2));
        assert_eq!(p.live(), 2);
    }

    #[test]
    fn removing_a_file_drops_its_postings_and_frees_slots() {
        let mut p = Postings::default();
        let a = chunks("a.rs", "alpha beta\n");
        let b = chunks("b.rs", "beta gamma\n");
        p.add_file("a.rs", &a);
        p.add_file("b.rs", &b);
        p.remove_file("a.rs", &a);
        assert!(p.postings("alpha").is_none(), "term only in a.rs is gone");
        assert_eq!(p.postings("beta").map(<[_]>::len), Some(1));
        assert_eq!(p.live(), 1);
        assert_eq!(p.avg_len(), f64::from(b[0].len), "only b.rs remains");
        // Re-adding reuses the freed slot instead of growing the table.
        let slots_before = p.slots.len();
        p.add_file("a.rs", &a);
        assert_eq!(p.slots.len(), slots_before);
        assert_eq!(p.live(), 2);
    }

    #[test]
    fn replacing_a_file_matches_a_fresh_build() {
        let mut replaced = Postings::default();
        let old = chunks("x.rs", "old words here\n");
        replaced.add_file("x.rs", &old);
        replaced.remove_file("x.rs", &old);
        let new = chunks("x.rs", "fresh words\n");
        replaced.add_file("x.rs", &new);

        let mut fresh = Postings::default();
        fresh.add_file("x.rs", &new);
        assert_eq!(replaced.live(), fresh.live());
        assert_eq!(replaced.avg_len(), fresh.avg_len());
        assert_eq!(replaced.term_count(), fresh.term_count());
        assert!(replaced.postings("old").is_none());
    }

    #[test]
    fn removing_unknown_file_is_a_no_op() {
        let mut p = Postings::default();
        p.remove_file("missing.rs", &[]);
        assert_eq!(p.live(), 0);
        assert_eq!(p.avg_len(), 0.0);
    }
}
