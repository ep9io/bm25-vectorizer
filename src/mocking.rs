//! This module contains simple tokenizers and indexers that are used for testing and examples and not designed for
//! real use cases.  Real use cases could involve performance optimisations and additional pre-processing steps such as:
//! stop word removal, stemming/lemmatisation, punctuation removal, n-grams, handling language specific features, etc.
//!  

use crate::bm25_token_indexer::Bm25TokenIndexer;
use crate::Bm25Tokenizer;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Simple whitespace tokenizer
pub struct MockWhitespaceTokenizer;

impl Bm25Tokenizer for MockWhitespaceTokenizer {
    fn tokenize(&self, input_text: &str) -> Vec<String> {
        input_text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
}

/// Case-preserving tokenizer
pub struct MockCasePreservingTokenizer;

impl Bm25Tokenizer for MockCasePreservingTokenizer {
    fn tokenize(&self, input_text: &str) -> Vec<String> {
        input_text.split_whitespace().map(String::from).collect()
    }
}

/// Punctuation-aware tokenizer
pub struct MockPunctuationTokenizer;

impl Bm25Tokenizer for MockPunctuationTokenizer {
    fn tokenize(&self, input_text: &str) -> Vec<String> {
        input_text
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
}

/// Hash-based token indexer
pub struct MockHashTokenIndexer;

impl Bm25TokenIndexer for MockHashTokenIndexer {
    type Bm25TokenIndex = u64;

    fn index(&self, token: &str) -> Self::Bm25TokenIndex {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        hasher.finish()
    }
}

/// Dictionary-based token indexer with interior mutability
pub struct MockDictionaryTokenIndexer {
    token_to_id: RefCell<HashMap<String, usize>>,
    next_id: RefCell<usize>,
}

impl MockDictionaryTokenIndexer {
    pub fn new() -> Self {
        Self {
            token_to_id: RefCell::new(HashMap::new()),
            next_id: RefCell::new(0),
        }
    }
}

impl Bm25TokenIndexer for MockDictionaryTokenIndexer {
    type Bm25TokenIndex = usize;

    fn index(&self, token: &str) -> Self::Bm25TokenIndex {
        let mut token_map = self.token_to_id.borrow_mut();
        let mut next_id = self.next_id.borrow_mut();

        if let Some(&id) = token_map.get(token) {
            id
        } else {
            let id = *next_id;
            token_map.insert(token.to_string(), id);
            *next_id += 1;
            id
        }
    }
}

/// String-based token indexer
pub struct MockStringTokenIndexer;

impl Bm25TokenIndexer for MockStringTokenIndexer {
    type Bm25TokenIndex = String;

    fn index(&self, token: &str) -> Self::Bm25TokenIndex {
        format!("idx_{}", token)
    }
}
