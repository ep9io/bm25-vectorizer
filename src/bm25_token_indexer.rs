/// Trait for mapping tokens to unique indices for efficient BM25 processing.
///
/// This trait defines how string tokens are converted to numerical or other
/// indexable representations.
///
/// Some indexing strategies include:
/// - **Hash-based**: Use hash functions (e.g. Murmur3) to map tokens to integers
/// - **Dictionary-based**: Maintain a mapping from tokens to sequential indices
///
/// # Type Parameters
///
/// * `Bm25TokenIndex` - The type used to represent token indices. This should
///   typically implement `Hash`, `Eq`, `Clone`, and other traits required for
///   use as map keys.
///
/// # Examples
///
/// ```rust
/// use bm25_vectorizer::Bm25TokenIndexer;
/// use std::collections::HashMap;
///
/// // Hash-based token indexer
/// struct HashTokenIndexer;
///
/// impl Bm25TokenIndexer for HashTokenIndexer {
///     type Bm25TokenIndex = u64;
///
///     fn index(&self, token: &str) -> Self::Bm25TokenIndex {
///         use std::hash::{Hash, Hasher};
///         // Note: Better hashing algorithms can be used (e.g. Murmur3)
///         use std::collections::hash_map::DefaultHasher;
///
///         let mut hasher = DefaultHasher::new();
///        token.hash(&mut hasher);
///         hasher.finish()
///     }
/// }
///
/// // Dictionary-based token indexer
/// struct DictionaryIndexer {
///     token_to_id: HashMap<String, usize>,
///     next_id: usize,
/// }
///
/// impl DictionaryIndexer {
///     fn new() -> Self {
///         Self {
///             token_to_id: HashMap::new(),
///             next_id: 0,
///         }
///     }
/// }
///
/// impl Bm25TokenIndexer for DictionaryIndexer {
///     type Bm25TokenIndex = usize;
///
///     fn index(&self, token: &str) -> Self::Bm25TokenIndex {
///         // Note: In a real implementation, you'd want interior mutability
///         // or a different API design to handle the mutable state
///         self.token_to_id.get(token).copied().unwrap_or(0)
///     }
/// }
/// ```
pub trait Bm25TokenIndexer {
    /// The type used to represent token indices.
    /// This associated type defines what kind of index representation is used
    /// for tokens.
    type Bm25TokenIndex;

    /// Maps a token string to its corresponding index representation.
    ///
    /// This method converts a string token into the index type defined by
    /// `Bm25TokenIndex`.
    ///
    /// # Arguments
    ///
    /// * `token` - The string token to be indexed
    ///
    /// # Returns
    ///
    /// An index of type `Self::Bm25TokenIndex` that uniquely represents the token
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bm25_vectorizer::Bm25TokenIndexer;
    /// use std::hash::{Hash, Hasher, DefaultHasher};
    ///
    /// struct HashIndexer;
    ///
    /// impl Bm25TokenIndexer for HashIndexer {
    ///     type Bm25TokenIndex = u64;
    ///
    ///     fn index(&self, token: &str) -> Self::Bm25TokenIndex {
    ///         let mut hasher = DefaultHasher::new();
    ///         token.hash(&mut hasher);
    ///         hasher.finish()
    ///     }
    /// }
    ///
    /// let indexer = HashIndexer;
    /// let index1 = indexer.index("hello");
    /// let index2 = indexer.index("hello");
    /// assert_eq!(index1, index2); // Same token, same index
    /// ```
    fn index(&self, token: &str) -> Self::Bm25TokenIndex;
}

// Tests for Bm25TokenIndexer trait
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mocking::{
        MockDictionaryTokenIndexer, MockHashTokenIndexer, MockStringTokenIndexer,
        MockWhitespaceTokenizer,
    };
    use crate::Bm25Tokenizer;

    #[test]
    fn test_hash_token_indexer_deterministic() {
        let indexer = MockHashTokenIndexer;
        let index1 = indexer.index("hello");
        let index2 = indexer.index("hello");
        assert_eq!(index1, index2, "Same token should produce same index");
    }

    #[test]
    fn test_hash_token_indexer_different_tokens() {
        let indexer = MockHashTokenIndexer;
        let index1 = indexer.index("hello");
        let index2 = indexer.index("world");
        assert_ne!(
            index1, index2,
            "Different tokens should produce different indices"
        );
    }

    #[test]
    fn test_hash_token_indexer_case_sensitivity() {
        let indexer = MockHashTokenIndexer;
        let index1 = indexer.index("hello");
        let index2 = indexer.index("Hello");
        assert_ne!(
            index1, index2,
            "Case-different tokens should produce different indices"
        );
    }

    #[test]
    fn test_dictionary_token_indexer_sequential() {
        let indexer = MockDictionaryTokenIndexer::new();
        let index1 = indexer.index("hello");
        let index2 = indexer.index("world");
        let index3 = indexer.index("rust");

        assert_eq!(index1, 0);
        assert_eq!(index2, 1);
        assert_eq!(index3, 2);
    }

    #[test]
    fn test_dictionary_token_indexer_deterministic() {
        let indexer = MockDictionaryTokenIndexer::new();
        let index1 = indexer.index("hello");
        let index2 = indexer.index("world");
        let index3 = indexer.index("hello"); // Repeat

        assert_eq!(index1, index3, "Same token should produce same index");
        assert_ne!(
            index1, index2,
            "Different tokens should produce different indices"
        );
    }

    #[test]
    fn test_dictionary_token_indexer_empty_string() {
        let indexer = MockDictionaryTokenIndexer::new();
        let index1 = indexer.index("");
        let index2 = indexer.index("");
        assert_eq!(
            index1, index2,
            "Empty string should be handled consistently"
        );
    }

    #[test]
    fn test_string_token_indexer() {
        let indexer = MockStringTokenIndexer;
        let index1 = indexer.index("hello");
        let index2 = indexer.index("world");

        assert_eq!(index1, "idx_hello");
        assert_eq!(index2, "idx_world");
    }

    #[test]
    fn test_string_token_indexer_deterministic() {
        let indexer = MockStringTokenIndexer;
        let index1 = indexer.index("test");
        let index2 = indexer.index("test");
        assert_eq!(index1, index2, "Same token should produce same index");
    }

    // Integration tests combining tokenizer and indexer

    #[test]
    fn test_tokenizer_indexer_integration() {
        let tokenizer = MockWhitespaceTokenizer;
        let indexer = MockHashTokenIndexer;

        let text = "hello world hello rust";
        let tokens = tokenizer.tokenize(text);
        let indices: Vec<u64> = tokens.iter().map(|token| indexer.index(token)).collect();

        // Should have 4 indices
        assert_eq!(indices.len(), 4);

        // "hello" appears twice and should have the same index
        assert_eq!(
            indices[0], indices[2],
            "Repeated token 'hello' should have same index"
        );

        // All other tokens should be different
        assert_ne!(
            indices[0], indices[1],
            "'hello' and 'world' should have different indices"
        );
        assert_ne!(
            indices[1], indices[3],
            "'world' and 'rust' should have different indices"
        );
        assert_ne!(
            indices[0], indices[3],
            "'hello' and 'rust' should have different indices"
        );
    }

    #[test]
    fn test_dictionary_indexer_with_tokenizer() {
        let tokenizer = MockWhitespaceTokenizer;
        let indexer = MockDictionaryTokenIndexer::new();

        let text = "the quick brown fox jumps over the lazy dog";
        let tokens = tokenizer.tokenize(text);
        let indices: Vec<usize> = tokens.iter().map(|token| indexer.index(token)).collect();

        // Should have 9 indices (same length as tokens)
        assert_eq!(indices.len(), 9);

        // "the" appears twice at positions 0 and 6, should have same index
        let the_index = indexer.index("the");
        assert_eq!(indices[0], the_index);
        assert_eq!(indices[6], the_index);
        assert_eq!(
            indices[0], indices[6],
            "Repeated token 'the' should have same index"
        );
    }

    #[test]
    fn test_edge_cases() {
        let tokenizer = MockWhitespaceTokenizer;
        let indexer = MockHashTokenIndexer;

        // Test with whitespace-only string
        let tokens = tokenizer.tokenize("   \t  \n  ");
        assert!(
            tokens.is_empty(),
            "Whitespace-only string should produce no tokens"
        );

        // Test with single character
        let tokens = tokenizer.tokenize("a");
        assert_eq!(tokens, vec!["a"]);
        let index = indexer.index(&tokens[0]);
        assert!(index > 0, "Single character should produce valid index");

        // Test with very long token
        let long_token = "a".repeat(1000);
        let index1 = indexer.index(&long_token);
        let index2 = indexer.index(&long_token);
        assert_eq!(index1, index2, "Long token should be handled consistently");
    }

    #[test]
    fn test_indexer_properties() {
        let indexer = MockHashTokenIndexer;

        // Property: same input should always produce same output
        let token = "consistent";
        let index1 = indexer.index(token);
        let index2 = indexer.index(token);
        assert_eq!(index1, index2, "Indexer should be deterministic");

        // Property: different inputs should generally produce different outputs
        // (Note: hash collisions are possible but rare)
        let index_a = indexer.index("a");
        let index_b = indexer.index("b");
        assert_ne!(
            index_a, index_b,
            "Different tokens should generally have different indices"
        );
    }
}
