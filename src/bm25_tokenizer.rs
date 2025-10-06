//! # BM25 Tokenization Traits
//!
//! This module defines the core tokenization and token indexing traits used by the BM25 vectorizer.
//!
//! The BM25 (Best Matching 25) algorithm requires two main operations:
//! 1. **Tokenization**: Breaking text into individual tokens (words, terms, etc.)
//! 2. **Token Indexing**: Mapping tokens to unique identifiers.
//!
//! These traits provide a flexible abstraction that allows users to implement custom
//! tokenization strategies (e.g., whitespace splitting, stemming, n-grams) and indexing
//! schemes (e.g., hash-based, etc) according to their specific needs.
//!
//! # Examples
//!
//! ```rust
//! use bm25_vectorizer::{Bm25Tokenizer, Bm25TokenIndexer};
//! use std::collections::HashMap;
//!
//! // Simple whitespace tokenizer
//! struct SimpleTokenizer;
//!
//! impl Bm25Tokenizer for SimpleTokenizer {
//!     fn tokenize(&self, input_text: &str) -> Vec<String> {
//!         input_text
//!             .split_whitespace()
//!             .map(|s| s.to_lowercase())
//!             .collect()
//!     }
//! }
//!
//! // Hash-based token indexer
//! struct HashTokenIndexer;
//!
//! impl Bm25TokenIndexer for HashTokenIndexer {
//!     type Bm25TokenIndex = u64;
//!
//!     fn index(&self, token: &str) -> Self::Bm25TokenIndex {
//!         use std::hash::{Hash, Hasher};
//!         // Note: Better hashing algorithms can be used (e.g. Murmur3)
//!         use std::collections::hash_map::DefaultHasher;
//!         
//!         let mut hasher = DefaultHasher::new();
//!         token.hash(&mut hasher);
//!         hasher.finish()
//!     }
//! }
//! ```

/// Trait for tokenizing text into individual terms for BM25 processing.
///
/// Implementors of this trait define how input text should be broken down into
/// individual tokens. This is an important step in the BM25 algorithm as it determines
/// how documents are analysed and indexed.
///
/// Common tokenization strategies include:
/// - **Whitespace splitting**: Split on spaces and punctuation
/// - **Stemming/Lemmatization**: Reduce words to their root forms
/// - **N-gram generation**: Create overlapping sequences of words
/// - **Language-specific processing**: Handle specific language features
///
/// # Examples
///
/// ```rust
/// use bm25_vectorizer::Bm25Tokenizer;
///
/// struct WhitespaceTokenizer;
///
/// impl Bm25Tokenizer for WhitespaceTokenizer {
///     fn tokenize(&self, input_text: &str) -> Vec<String> {
///         input_text
///             .split_whitespace()
///             .map(|token| token.to_lowercase())
///             .collect()
///     }
/// }
///
/// let tokenizer = WhitespaceTokenizer;
/// let tokens = tokenizer.tokenize("Hello World Example");
/// assert_eq!(tokens, vec!["hello", "world", "example"]);
/// ```
pub trait Bm25Tokenizer {
    /// Tokenizes the input text into a vector of string tokens.
    ///
    /// This method takes a string slice and returns a vector of tokens that will
    /// be used for BM25 scoring.
    ///
    /// # Arguments
    ///
    /// * `input_text` - The text to be tokenized
    ///
    /// # Returns
    ///
    /// A vector of string tokens extracted from the input text
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bm25_vectorizer::Bm25Tokenizer;
    ///
    /// struct SimpleTokenizer;
    /// impl Bm25Tokenizer for SimpleTokenizer {
    ///     fn tokenize(&self, input_text: &str) -> Vec<String> {
    ///         input_text.split_whitespace()
    ///                   .map(String::from)
    ///                   .collect()
    ///     }
    /// }
    ///
    /// let tokenizer = SimpleTokenizer;
    /// let tokens = tokenizer.tokenize("rust is awesome");
    /// assert_eq!(tokens, vec!["rust", "is", "awesome"]);
    /// ```
    fn tokenize(&self, input_text: &str) -> Vec<String>;
}

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

// Mock implementations for testing and examples
use std::cell::RefCell;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

/// Simple whitespace tokenizer for testing and examples
pub struct MockWhitespaceTokenizer;

impl Bm25Tokenizer for MockWhitespaceTokenizer {
    fn tokenize(&self, input_text: &str) -> Vec<String> {
        input_text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
}

/// Case-preserving tokenizer for testing and examples
pub struct MockCasePreservingTokenizer;

impl Bm25Tokenizer for MockCasePreservingTokenizer {
    fn tokenize(&self, input_text: &str) -> Vec<String> {
        input_text.split_whitespace().map(String::from).collect()
    }
}

/// Punctuation-aware tokenizer for testing and examples
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

/// Hash-based token indexer for testing and examples
pub struct MockHashTokenIndexer;

impl Bm25TokenIndexer for MockHashTokenIndexer {
    type Bm25TokenIndex = u64;

    fn index(&self, token: &str) -> Self::Bm25TokenIndex {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        hasher.finish()
    }
}

/// Dictionary-based token indexer with interior mutability for testing and examples
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

/// String-based token indexer for testing and examples
pub struct MockStringTokenIndexer;

impl Bm25TokenIndexer for MockStringTokenIndexer {
    type Bm25TokenIndex = String;

    fn index(&self, token: &str) -> Self::Bm25TokenIndex {
        format!("idx_{}", token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for Bm25Tokenizer trait

    #[test]
    fn test_whitespace_tokenizer_basic() {
        let tokenizer = MockWhitespaceTokenizer;
        let tokens = tokenizer.tokenize("hello world rust");
        assert_eq!(tokens, vec!["hello", "world", "rust"]);
    }

    #[test]
    fn test_whitespace_tokenizer_case_normalization() {
        let tokenizer = MockWhitespaceTokenizer;
        let tokens = tokenizer.tokenize("Hello WORLD RusT");
        assert_eq!(tokens, vec!["hello", "world", "rust"]);
    }

    #[test]
    fn test_whitespace_tokenizer_empty_string() {
        let tokenizer = MockWhitespaceTokenizer;
        let tokens = tokenizer.tokenize("");
        assert_eq!(tokens, Vec::<String>::new());
    }

    #[test]
    fn test_whitespace_tokenizer_single_token() {
        let tokenizer = MockWhitespaceTokenizer;
        let tokens = tokenizer.tokenize("hello");
        assert_eq!(tokens, vec!["hello"]);
    }

    #[test]
    fn test_whitespace_tokenizer_multiple_spaces() {
        let tokenizer = MockWhitespaceTokenizer;
        let tokens = tokenizer.tokenize("hello    world   rust");
        assert_eq!(tokens, vec!["hello", "world", "rust"]);
    }

    #[test]
    fn test_whitespace_tokenizer_leading_trailing_spaces() {
        let tokenizer = MockWhitespaceTokenizer;
        let tokens = tokenizer.tokenize("  hello world  ");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_case_preserving_tokenizer() {
        let tokenizer = MockCasePreservingTokenizer;
        let tokens = tokenizer.tokenize("Hello WORLD RusT");
        assert_eq!(tokens, vec!["Hello", "WORLD", "RusT"]);
    }

    #[test]
    fn test_punctuation_tokenizer() {
        let tokenizer = MockPunctuationTokenizer;
        let tokens = tokenizer.tokenize("hello, world! rust?");
        assert_eq!(tokens, vec!["hello", "world", "rust"]);
    }

    #[test]
    fn test_punctuation_tokenizer_numbers() {
        let tokenizer = MockPunctuationTokenizer;
        let tokens = tokenizer.tokenize("version 2.0 is great!");
        assert_eq!(tokens, vec!["version", "20", "is", "great"]);
    }

    // Tests for Bm25TokenIndexer trait

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

    // Property-based style tests

    #[test]
    fn test_tokenizer_properties() {
        let tokenizer = MockWhitespaceTokenizer;

        // Property: tokenizing empty string should return empty vector
        assert!(tokenizer.tokenize("").is_empty());

        // Property: tokenizing single word should return vector with one element
        let result = tokenizer.tokenize("word");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "word");

        // Property: all tokens should be lowercase (for this specific tokenizer)
        let result = tokenizer.tokenize("HELLO World");
        for token in &result {
            assert_eq!(
                token.to_lowercase(),
                *token,
                "All tokens should be lowercase"
            );
        }
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
