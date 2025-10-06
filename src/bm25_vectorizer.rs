//! # BM25 Vectorizer
//!
//! This module implements the BM25 (Best Matching 25) ranking function, a probabilistic 
//! ranking algorithm commonly used in information retrieval and search engines.
//!
//! ## BM25 Algorithm
//!
//! BM25 is a bag-of-words retrieval function that ranks matching documents by their relevance
//! to a query.
//!
//! The BM25 value for a term in a document is calculated as:
//!
//! ```text
//! BM25(t,d) = (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * (|d| / avgdl)))
//! ```
//!
//! The BM25+ value for a term in a document is calculated as:
//!
//! ```text
//! BM25(t,d) = (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * (|d| / avgdl))) + δ
//! ```
//!
//! Where:
//! - `tf(t,d)` is the term frequency in the document
//! - `k1` controls term frequency saturation (typically 1.2)
//! - `b` controls length normalisation (typically 0.75)
//! - `|d|` is the document length
//! - `avgdl` is the average document length in the corpus
//! - `δ` (delta) is a lower bound for term frequency scoring
//!
//! ## Usage
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use bm25_vectorizer::{Bm25VectorizerBuilder, MockWhitespaceTokenizer, MockHashTokenIndexer};
//!
//! let corpus = vec!["hello world", "world of rust", "hello rust"];
//! let tokenizer = MockWhitespaceTokenizer;
//! let indexer = MockHashTokenIndexer;
//!
//! let vectorizer = Bm25VectorizerBuilder::new()
//!     .tokenizer(tokenizer)
//!     .token_indexer(indexer)
//!     .k1(1.2)
//!     .b(0.75)
//!     .fit(&corpus)?
//!     .build()?;
//!
//! let vector = vectorizer.vectorize("hello world");
//! # Ok(())
//! # }
//! ```

use crate::bm25_tokenizer::{Bm25TokenIndexer, Bm25Tokenizer};
use crate::bm25_vectorizer::Bm25VectorizerError::{
    InvalidAverageDocumentLength, InvalidTermFrequencyLowerBound, InvalidTermRelevanceSaturation,
    MissingAverageDocumentLength, MissingTokenIndexer, MissingTokenizer,
};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;

#[cfg(feature = "parallelism")]
use rayon::prelude::*;

/// Represents a token with its index and BM25 value.
///
/// # Type Parameters
/// - `T`: The type of the token index (must implement required traits for the vectorizer)
///
/// # Fields
/// - `index`: The token's unique identifier
/// - `value`: The computed BM25 value for this token
#[derive(PartialEq, Debug, Clone, PartialOrd)]
pub struct TokenIndexValue<T> {
    /// The unique identifier for the token
    pub index: T,
    /// The BM25 value value for this token
    pub value: f32,
}

/// A sparse vector representation containing token indices and their BM25 values.
///
/// # Type Parameters
/// - `T`: The type of token indices used
///
/// # Examples
/// ```rust
/// use bm25_vectorizer::{TokenIndexValue, SparseRepresentation};
/// 
/// let tokens = vec![
///     TokenIndexValue { index: 0, value: 1.2 },
///     TokenIndexValue { index: 5, value: 0.8 },
/// ];
/// let sparse_vector = SparseRepresentation(tokens);
/// ```
#[derive(PartialEq, Debug, Clone, PartialOrd)]
pub struct SparseRepresentation<T>(pub Vec<TokenIndexValue<T>>);

/// Controls term frequency saturation.
#[derive(Debug)]
pub struct TermRelevanceSaturation {
    k1: f32,
}

/// The additional δ parameter for BM25+
#[derive(Debug)]
pub struct TermFrequencyLowerBound {
    delta: f32,
}

/// Controls document length normalisation.
#[derive(Debug)]
pub struct LengthNormalisation {
    b: f32,
}

/// Represents the average document length in the corpus.
///
/// This value is used for document length normalisation.
/// It's typically computed during the fitting process by analysing the training corpus.
#[derive(Debug)]
pub struct AverageDocumentLength {
    avgdl: f32,
}

/// The main BM25 vectorizer that converts text into sparse vector representations.
///
/// This struct encapsulates all the parameters and components needed to perform BM25 
/// vectorization. It uses a tokenizer to break text into tokens and a token indexer 
/// to map tokens to indices.
///
/// # Type Parameters
/// - `TokenIndexer`: Implementation of `Bm25TokenIndexer` trait for mapping tokens to indices
/// - `Tokenizer`: Implementation of `Bm25Tokenizer` trait for text tokenization
///
/// # Examples
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use bm25_vectorizer::{Bm25VectorizerBuilder, MockWhitespaceTokenizer, MockHashTokenIndexer};
///
/// let corpus = vec!["hello world", "world of rust"];
/// let vectorizer = Bm25VectorizerBuilder::new()
///     .tokenizer(MockWhitespaceTokenizer)
///     .token_indexer(MockHashTokenIndexer)
///     .fit(&corpus)?
///     .build()?;
///
/// let result = vectorizer.vectorize("hello rust");
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Bm25Vectorizer<TokenIndexer, Tokenizer> {
    tokenizer: Tokenizer,
    k1: TermRelevanceSaturation,
    b: LengthNormalisation,
    avgdl: AverageDocumentLength,
    delta: TermFrequencyLowerBound,
    token_indexer: TokenIndexer,
}

impl<TokenIndexer, Tokenizer> Bm25Vectorizer<TokenIndexer, Tokenizer> {
    /// Returns the average document length used for normalisation.
    ///
    /// # Examples
    /// ```rust
    /// # use bm25_vectorizer::*;
    /// # let vectorizer = Bm25VectorizerBuilder::new()
    /// #     .tokenizer(MockWhitespaceTokenizer)
    /// #     .token_indexer(MockHashTokenIndexer)
    /// #     .avgdl(10.5)
    /// #     .build().unwrap();
    /// assert_eq!(vectorizer.avgdl(), 10.5);
    /// ```
    pub fn avgdl(&self) -> f32 {
        self.avgdl.avgdl
    }

    /// Returns the k1 parameter controlling term frequency saturation.
    ///
    /// # Examples
    /// ```rust
    /// # use bm25_vectorizer::*;
    /// # let vectorizer = Bm25VectorizerBuilder::new()
    /// #     .tokenizer(MockWhitespaceTokenizer)
    /// #     .token_indexer(MockHashTokenIndexer)
    /// #     .k1(1.5)
    /// #     .avgdl(10.0)
    /// #     .build().unwrap();
    /// assert_eq!(vectorizer.k1(), 1.5);
    /// ```
    pub fn k1(&self) -> f32 {
        self.k1.k1
    }

    /// Returns the b parameter controlling length normalisation.
    ///
    /// # Examples
    /// ```rust
    /// # use bm25_vectorizer::*;
    /// # let vectorizer = Bm25VectorizerBuilder::new()
    /// #     .tokenizer(MockWhitespaceTokenizer)
    /// #     .token_indexer(MockHashTokenIndexer)
    /// #     .b(0.8)
    /// #     .avgdl(10.0)
    /// #     .build().unwrap();
    /// assert_eq!(vectorizer.b(), 0.8);
    /// ```
    pub fn b(&self) -> f32 {
        self.b.b
    }

    /// Returns the delta parameter used as a lower bound for term values.
    ///
    /// # Examples
    /// ```rust
    /// # use bm25_vectorizer::*;
    /// # let vectorizer = Bm25VectorizerBuilder::new()
    /// #     .tokenizer(MockWhitespaceTokenizer)
    /// #     .token_indexer(MockHashTokenIndexer)
    /// #     .delta(0.25)
    /// #     .avgdl(10.0)
    /// #     .build().unwrap();
    /// assert_eq!(vectorizer.delta(), 0.25);
    /// ```
    pub fn delta(&self) -> f32 {
        self.delta.delta
    }

    /// Converts input text into a sparse BM25 vector representation.
    ///
    /// This method tokenizes the input text, computes term frequencies, and applies
    /// the BM25 to generate a sparse vector representation that can then be uploaded to a vector database.
    ///
    /// # Arguments
    /// - `text`: The input text to vectorize
    ///
    /// # Returns
    /// A `SparseRepresentation` containing token indices and their BM25 values
    ///
    /// # Examples
    /// ```rust
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use bm25_vectorizer::{Bm25VectorizerBuilder, MockWhitespaceTokenizer, MockHashTokenIndexer};
    ///
    /// let corpus = vec!["hello world", "world rust"];
    /// let vectorizer = Bm25VectorizerBuilder::new()
    ///     .tokenizer(MockWhitespaceTokenizer)
    ///     .token_indexer(MockHashTokenIndexer)
    ///     .fit(&corpus)?
    ///     .build()?;
    ///
    /// let result = vectorizer.vectorize("hello world");
    /// // Result contains BM25 values for tokens "hello" and "world"
    /// assert_eq!(result.0.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn vectorize(&self, text: &str) -> SparseRepresentation<TokenIndexer::Bm25TokenIndex>
    where
        TokenIndexer: Bm25TokenIndexer,
        TokenIndexer::Bm25TokenIndex: Eq + Hash + Clone + Debug + Ord,
        Tokenizer: Bm25Tokenizer,
    {
        let tokens = self.tokenizer.tokenize(text);
        let doc_length = tokens.len() as f32;

        // Build unique map of indices to their term frequencies
        // Using tree map for deterministic results
        let mut index_counts: BTreeMap<TokenIndexer::Bm25TokenIndex, usize> = BTreeMap::new();

        for token in tokens.iter() {
            let index = self.token_indexer.index(token);
            *index_counts.entry(index).or_insert(0) += 1;
        }

        let embeddings: Vec<TokenIndexValue<TokenIndexer::Bm25TokenIndex>> = index_counts
            .into_iter()
            .map(|(index, count)| {
                let token_frequency = count as f32;
                let numerator = token_frequency * (self.k1() + 1.0);
                let denominator = token_frequency
                    + self.k1() * (1.0 - self.b() + self.b() * (doc_length / self.avgdl()));

                let value = (numerator / denominator) + self.delta();

                TokenIndexValue { index, value }
            })
            .collect();

        SparseRepresentation(embeddings)
    }
}

/// Builder for creating and configuring a `Bm25Vectorizer`.
///
/// It supports fitting on a corpus to automatically compute the
/// average document length, and validates all parameters before building.
///
/// # Type Parameters
/// - `TokenIndexer`: Implementation of `Bm25TokenIndexer` trait
/// - `Tokenizer`: Implementation of `Bm25Tokenizer` trait
///
/// # Examples
///
/// Basic usage with manual avgdl:
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use bm25_vectorizer::{Bm25VectorizerBuilder, MockWhitespaceTokenizer, MockHashTokenIndexer};
///
/// let vectorizer = Bm25VectorizerBuilder::new()
///     .tokenizer(MockWhitespaceTokenizer)
///     .token_indexer(MockHashTokenIndexer)
///     .k1(1.2)
///     .b(0.75)
///     .avgdl(10.0)
///     .build()?;
/// # Ok(())
/// # }
/// ```
///
/// Usage with corpus fitting:
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use bm25_vectorizer::{Bm25VectorizerBuilder, MockWhitespaceTokenizer, MockHashTokenIndexer};
///
/// let corpus = vec!["hello world", "world of rust", "hello rust programming"];
/// let vectorizer = Bm25VectorizerBuilder::new()
///     .tokenizer(MockWhitespaceTokenizer)
///     .token_indexer(MockHashTokenIndexer)
///     .k1(1.2)
///     .b(0.75)
///     .fit(&corpus)?  // Automatically computes avgdl
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct Bm25VectorizerBuilder<TokenIndexer, Tokenizer> {
    tokenizer: Option<Tokenizer>,
    k1: TermRelevanceSaturation,
    b: LengthNormalisation,
    avgdl: Option<AverageDocumentLength>,
    delta: TermFrequencyLowerBound,
    token_indexer: Option<TokenIndexer>,
}

impl<TokenIndexer, Tokenizer> Bm25VectorizerBuilder<TokenIndexer, Tokenizer> {
    pub fn new() -> Self {
        Self {
            tokenizer: None,
            k1: TermRelevanceSaturation { k1: 1.2 },
            b: LengthNormalisation { b: 0.75 },
            avgdl: None,
            delta: TermFrequencyLowerBound { delta: 0.0 },
            token_indexer: None,
        }
    }

    pub fn k1(mut self, k1: f32) -> Self {
        self.k1 = TermRelevanceSaturation { k1 };
        self
    }

    pub fn b(mut self, b: f32) -> Self {
        self.b = LengthNormalisation { b };
        self
    }

    pub fn delta(mut self, delta: f32) -> Self {
        self.delta = TermFrequencyLowerBound { delta };
        self
    }

    pub fn avgdl(mut self, avgdl: f32) -> Self {
        self.avgdl = Some(AverageDocumentLength { avgdl });
        self
    }

    pub fn tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn token_indexer(mut self, token_indexer: TokenIndexer) -> Self {
        self.token_indexer = Some(token_indexer);
        self
    }

    pub fn fit(mut self, corpus: &[&str]) -> Result<Self, Bm25VectorizerError>
    where
        Tokenizer: Bm25Tokenizer + Sync,
    {
        if let Some(ref tokenizer) = self.tokenizer {
            let doc_count = corpus.len();
            if doc_count == 0 {
                return Err(Bm25VectorizerError::EmptyCorpus);
            }

            #[cfg(not(feature = "parallelism"))]
            let corpus_iter = corpus.iter();
            #[cfg(feature = "parallelism")]
            let corpus_iter = corpus.par_iter();

            let total_length: usize = corpus_iter.map(|doc| tokenizer.tokenize(doc).len()).sum();
            self.avgdl = Some(AverageDocumentLength {
                avgdl: total_length as f32 / doc_count as f32,
            });
        }
        Ok(self)
    }

    #[cfg(not(feature = "parallelism"))]
    pub fn fit_iter<I, S>(mut self, corpus: I) -> Result<Self, Bm25VectorizerError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
        Tokenizer: Bm25Tokenizer + Sync,
    {
        if let Some(ref tokenizer) = self.tokenizer {
            let (doc_count, total_length) = corpus
                .into_iter()
                .map(|doc| tokenizer.tokenize(doc.as_ref()).len())
                .fold((0usize, 0usize), |(count, sum), len| (count + 1, sum + len));

            self.avgdl = Some(AverageDocumentLength {
                avgdl: total_length as f32 / doc_count as f32,
            });
        }
        Ok(self)
    }

    #[cfg(feature = "parallelism")]
    pub fn fit_par_iter<I, S>(mut self, corpus: I) -> Result<Self, Bm25VectorizerError>
    where
        I: IntoIterator<Item = S>,
        I::IntoIter: Send,
        S: AsRef<str> + Send,
        Tokenizer: Bm25Tokenizer + Sync,
    {
        if let Some(ref tokenizer) = self.tokenizer {
            let (doc_count, total_length) = {
                use rayon::iter::ParallelBridge;
                corpus
                    .into_iter()
                    .par_bridge()
                    .map(|doc| tokenizer.tokenize(doc.as_ref()).len())
                    .fold(|| (0usize, 0usize), |(count, sum), len| (count + 1, sum + len))
                    .reduce(|| (0, 0), |(c1, s1), (c2, s2)| (c1 + c2, s1 + s2))
            };

            if doc_count == 0 {
                return Err(Bm25VectorizerError::EmptyCorpus);
            }

            self.avgdl = Some(AverageDocumentLength {
                avgdl: total_length as f32 / doc_count as f32,
            });
        }
        Ok(self)
    }


    pub fn build(self) -> Result<Bm25Vectorizer<TokenIndexer, Tokenizer>, Bm25VectorizerError> {
        let tokenizer = self.tokenizer.ok_or(MissingTokenizer)?;
        let token_indexer = self.token_indexer.ok_or(MissingTokenIndexer)?;
        let avgdl = self.avgdl.ok_or(MissingAverageDocumentLength)?;

        if &self.k1.k1 < &0.0 {
            return Err(InvalidTermRelevanceSaturation);
        }
        if &self.b.b < &0.0 || &self.b.b > &1.0 {
            return Err(InvalidTermRelevanceSaturation);
        }
        if &avgdl.avgdl <= &0.0 {
            return Err(InvalidAverageDocumentLength);
        }
        if &self.delta.delta < &0.0 {
            return Err(InvalidTermFrequencyLowerBound);
        }

        Ok(Bm25Vectorizer {
            tokenizer,
            k1: self.k1,
            b: self.b,
            avgdl,
            delta: self.delta,
            token_indexer,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Bm25VectorizerError {
    #[error("Cannot fit on empty corpus.")]
    EmptyCorpus,
    #[error("Average document length must be provided or computed via fit().")]
    MissingAverageDocumentLength,
    #[error("Tokenizer must be provided.")]
    MissingTokenizer,
    #[error("Token indexer must be provided.")]
    MissingTokenIndexer,
    #[error("Invalid b value: must be between 0 and 1.")]
    InvalidLengthNormalisation,
    #[error(
        "Invalid k1 value: should normally fall within the 0 to 3 range. However, there is no strict enforcement preventing values higher than 3."
    )]
    InvalidTermRelevanceSaturation,
    #[error("Invalid average document length: value must be greater than 0.")]
    InvalidAverageDocumentLength,
    #[error("Invalid delta (δ) value: must be 0 or greater.")]
    InvalidTermFrequencyLowerBound,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25_tokenizer::{MockWhitespaceTokenizer, MockHashTokenIndexer, MockDictionaryTokenIndexer};

    #[test]
    fn test_builder_new_defaults() {
        let builder = Bm25VectorizerBuilder::<MockHashTokenIndexer, MockWhitespaceTokenizer>::new();
        
        // Check default values
        assert_eq!(builder.k1.k1, 1.2);
        assert_eq!(builder.b.b, 0.75);
        assert_eq!(builder.delta.delta, 0.0);
        assert!(builder.tokenizer.is_none());
        assert!(builder.token_indexer.is_none());
        assert!(builder.avgdl.is_none());
    }

    #[test]
    fn test_builder_parameter_setting() {
        let builder = Bm25VectorizerBuilder::<MockHashTokenIndexer, MockWhitespaceTokenizer>::new()
            .k1(2.0)
            .b(0.5)
            .delta(0.25)
            .avgdl(15.0);

        assert_eq!(builder.k1.k1, 2.0);
        assert_eq!(builder.b.b, 0.5);
        assert_eq!(builder.delta.delta, 0.25);
        assert_eq!(builder.avgdl.unwrap().avgdl, 15.0);
    }

    #[test]
    fn test_builder_missing_components() {
        let result = Bm25VectorizerBuilder::<MockHashTokenIndexer, MockWhitespaceTokenizer>::new()
            .avgdl(10.0)
            .build();
        
        assert!(matches!(result, Err(MissingTokenizer)));

        let result = Bm25VectorizerBuilder::<MockHashTokenIndexer, MockWhitespaceTokenizer>::new()
            .tokenizer(MockWhitespaceTokenizer)
            .avgdl(10.0)
            .build();
        
        assert!(matches!(result, Err(MissingTokenIndexer)));

        let result = Bm25VectorizerBuilder::<MockHashTokenIndexer, MockWhitespaceTokenizer>::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .build();
        
        assert!(matches!(result, Err(MissingAverageDocumentLength)));
    }

    #[test]
    fn test_builder_invalid_parameters() {
        // Test negative k1
        let result = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .k1(-1.0)
            .avgdl(10.0)
            .build();
        
        assert!(matches!(result, Err(InvalidTermRelevanceSaturation)));

        // Test invalid b values
        let result = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .b(-0.1)
            .avgdl(10.0)
            .build();
        
        assert!(matches!(result, Err(InvalidTermRelevanceSaturation)));

        let result = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .b(1.1)
            .avgdl(10.0)
            .build();
        
        assert!(matches!(result, Err(InvalidTermRelevanceSaturation)));

        // Test invalid avgdl
        let result = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .avgdl(0.0)
            .build();
        
        assert!(matches!(result, Err(InvalidAverageDocumentLength)));

        // Test negative delta
        let result = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .delta(-0.1)
            .avgdl(10.0)
            .build();
        
        assert!(matches!(result, Err(InvalidTermFrequencyLowerBound)));
    }

    #[test]
    fn test_successful_build() {
        let vectorizer = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .k1(1.5)
            .b(0.8)
            .delta(0.25)
            .avgdl(12.0)
            .build()
            .unwrap();

        assert_eq!(vectorizer.k1(), 1.5);
        assert_eq!(vectorizer.b(), 0.8);
        assert_eq!(vectorizer.delta(), 0.25);
        assert_eq!(vectorizer.avgdl(), 12.0);
    }

    #[test]
    fn test_fit_corpus() {
        let corpus = vec!["hello world", "world of rust", "hello rust programming"];
        let builder = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .fit(&corpus)
            .unwrap();

        // Average document length should be (2 + 3 + 3) / 3 = 2.67 (approximately)
        let expected_avgdl = (2.0 + 3.0 + 3.0) / 3.0;
        assert_eq!(builder.avgdl.unwrap().avgdl, expected_avgdl);
    }

    #[test]
    fn test_fit_empty_corpus() {
        let corpus: Vec<&str> = vec![];
        let result = Bm25VectorizerBuilder::<MockHashTokenIndexer, MockWhitespaceTokenizer>::new()
            .tokenizer(MockWhitespaceTokenizer)
            .fit(&corpus);

        assert!(matches!(result, Err(Bm25VectorizerError::EmptyCorpus)));
    }

    #[test]
    fn test_vectorize_basic() {
        let vectorizer = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockDictionaryTokenIndexer::new())
            .avgdl(2.0)
            .build()
            .unwrap();

        let result = vectorizer.vectorize("hello world");
        
        // Should have 2 tokens
        assert_eq!(result.0.len(), 2);
        
        // All values should be positive due to BM25 formula
        for token in &result.0 {
            assert!(token.value > 0.0);
        }
    }

    #[test]
    fn test_vectorize_repeated_tokens() {
        let vectorizer = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockDictionaryTokenIndexer::new())
            .avgdl(3.0)
            .build()
            .unwrap();

        let result = vectorizer.vectorize("hello hello world");
        
        // Should have 2 unique tokens (hello appears twice, world once)
        assert_eq!(result.0.len(), 2);
        
        // Token for "hello" should have higher value due to higher frequency
        let hello_value = result.0.iter().find(|t| t.index == 0).unwrap().value;  // "hello" gets index 0
        let world_value = result.0.iter().find(|t| t.index == 1).unwrap().value;  // "world" gets index 1
        
        assert!(hello_value > world_value);
    }

    #[test]
    fn test_vectorize_empty_text() {
        let vectorizer = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockHashTokenIndexer)
            .avgdl(2.0)
            .build()
            .unwrap();

        let result = vectorizer.vectorize("");
        assert_eq!(result.0.len(), 0);
    }

    #[test]
    fn test_bm25_parameters_effect() {
        // Test that changing k1 affects the values
        let vectorizer_low_k1 = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockDictionaryTokenIndexer::new())
            .k1(0.5)
            .avgdl(2.0)
            .build()
            .unwrap();

        let vectorizer_high_k1 = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockDictionaryTokenIndexer::new())
            .k1(3.0)
            .avgdl(2.0)
            .build()
            .unwrap();

        let result_low = vectorizer_low_k1.vectorize("hello hello");
        let result_high = vectorizer_high_k1.vectorize("hello hello");

        // Higher k1 should result in higher values for repeated terms
        assert!(result_high.0[0].value > result_low.0[0].value);
    }

    #[test]
    fn test_length_normalisation_effect() {
        let vectorizer_no_norm = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockDictionaryTokenIndexer::new())
            .b(0.0)  // No length normalisation
            .avgdl(5.0)
            .build()
            .unwrap();

        let vectorizer_full_norm = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockDictionaryTokenIndexer::new())
            .b(1.0)  // Full length normalisation
            .avgdl(5.0)
            .build()
            .unwrap();

        // Test with a longer document
        let long_text = "hello world this is a long document";
        let short_text = "hello world";

        let long_no_norm = vectorizer_no_norm.vectorize(long_text);
        let long_full_norm = vectorizer_full_norm.vectorize(long_text);
        let short_no_norm = vectorizer_no_norm.vectorize(short_text);

        // With no normalisation, longer docs don't get penalised as much
        // With full normalisation, values should be more similar between docs
        let hello_long_no_norm = long_no_norm.0.iter().find(|t| t.index == 0).unwrap().value;
        let hello_long_full_norm = long_full_norm.0.iter().find(|t| t.index == 0).unwrap().value;
        let hello_short_no_norm = short_no_norm.0.iter().find(|t| t.index == 0).unwrap().value;

        // Length normalisation should make long document values lower
        assert!(hello_long_no_norm > hello_long_full_norm);
        assert!(hello_short_no_norm > hello_long_full_norm);
    }

    #[test]
    fn test_delta_effect() {
        let vectorizer_no_delta = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockDictionaryTokenIndexer::new())
            .delta(0.0)
            .avgdl(2.0)
            .build()
            .unwrap();

        let vectorizer_with_delta = Bm25VectorizerBuilder::new()
            .tokenizer(MockWhitespaceTokenizer)
            .token_indexer(MockDictionaryTokenIndexer::new())
            .delta(0.5)
            .avgdl(2.0)
            .build()
            .unwrap();

        let result_no_delta = vectorizer_no_delta.vectorize("hello");
        let result_with_delta = vectorizer_with_delta.vectorize("hello");

        // Delta should add to all values
        assert_eq!(result_with_delta.0[0].value, result_no_delta.0[0].value + 0.5);
    }

    #[cfg(not(feature = "parallelism"))]
    #[test]
    fn test_fit_iter() {
        let corpus = vec!["hello world", "world rust", "hello programming"];
        let builder = Bm25VectorizerBuilder::<MockHashTokenIndexer, MockWhitespaceTokenizer>::new()
            .tokenizer(MockWhitespaceTokenizer)
            .fit_iter(corpus)
            .unwrap();

        let expected_avgdl = (2.0 + 2.0 + 2.0) / 3.0;
        assert_eq!(builder.avgdl.unwrap().avgdl, expected_avgdl);
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn test_fit_par_iter() {
        let corpus = vec!["hello world", "world rust", "hello programming"];
        let builder = Bm25VectorizerBuilder::<MockHashTokenIndexer, MockWhitespaceTokenizer>::new()
            .tokenizer(MockWhitespaceTokenizer)
            .fit_par_iter(corpus)
            .unwrap();

        let expected_avgdl = (2.0 + 2.0 + 2.0) / 3.0;
        assert_eq!(builder.avgdl.unwrap().avgdl, expected_avgdl);
    }
}
