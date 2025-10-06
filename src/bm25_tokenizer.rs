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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mocking::{
        MockCasePreservingTokenizer,
        MockPunctuationTokenizer, MockWhitespaceTokenizer,
    };
    use rust_stemmers::{Algorithm as StemmingAlgorithm, Stemmer};
    use stop_words::{get, LANGUAGE as StopWordLanguage};
    use unicode_segmentation::UnicodeSegmentation;

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

    /// Example tokenizer.
    /// Performs: Unicode normalisation → lowercase → tokenisation → stop word removal → stemming
    struct SampleNlpTokenizer;

    impl SampleNlpTokenizer {
        fn new() -> Self {
            Self
        }
    }

    impl Bm25Tokenizer for SampleNlpTokenizer {
        /// Tokenizes input text through a NLP pipeline.
        ///
        /// # Processing Steps
        ///
        /// 1. **Unicode Normalisation**: Converts non-ASCII characters to ASCII equivalents
        /// 2. **Lowercase Conversion**: Ensures case-insensitive matching
        /// 3. **Word Segmentation**: Splits text into tokens using Unicode word boundaries
        /// 4. **Stop Word Removal**: Filters out common words (e.g., "the", "is", "at")
        /// 5. **Stemming**: Reduces words to their root form (e.g., "running" → "run")
        fn tokenize(&self, input_text: &str) -> Vec<String> {
            // Step 1: Normalise Unicode characters to ASCII
            // U+FFFD � REPLACEMENT CHARACTER used to replace an unknown, unrecognised, or unrepresentable character
            let text = deunicode::deunicode_with_tofu_cow(input_text, "�");

            // Step 2: Convert to lowercase for consistent processing
            let text = text.to_lowercase();

            // Step 3: Tokenise into words using Unicode segmentation
            let tokens: Vec<&str> = text
                .unicode_words()
                .filter(|word| !word.is_empty())
                .collect();

            // Step 4 & 5: Remove stop words and apply stemming
            let stop_words = get(StopWordLanguage::English);
            let stemmer = Stemmer::create(StemmingAlgorithm::English);

            tokens
                .into_iter()
                .filter(|token| !stop_words.contains(&*token))
                .map(|token| stemmer.stem(token).to_string())
                .collect()
        }
    }

    #[test]
    fn test_nlp_tokenizer_basic() {
        let tokenizer = SampleNlpTokenizer::new();
        let tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog");
        // Tokens:
        // [0] = {alloc::string::String} "quick"
        // [1] = {alloc::string::String} "brown"
        // [2] = {alloc::string::String} "fox"
        // [3] = {alloc::string::String} "jump"
        // [4] = {alloc::string::String} "lazi"

        // Should not contain stop words
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"over".to_string()));

        // Should contain stemmed content words
        assert!(tokens.iter().any(|t| t.starts_with("quick")));
        assert!(tokens.iter().any(|t| t.starts_with("jump")));
    }

    #[test]
    fn test_nlp_tokenizer_pipeline() {
        let tokenizer = SampleNlpTokenizer::new();
        let input_text = "Modern computing owes much to the theoretical foundations laid by pioneers in mathematics and logic.";

        let tokens = tokenizer.tokenize(input_text);
        // Tokens:
        // [0] = {alloc::string::String} "modern"
        // [1] = {alloc::string::String} "comput"
        // [2] = {alloc::string::String} "owe"
        // [3] = {alloc::string::String} "theoret"
        // [4] = {alloc::string::String} "foundat"
        // [5] = {alloc::string::String} "laid"
        // [6] = {alloc::string::String} "pioneer"
        // [7] = {alloc::string::String} "mathemat"
        // [8] = {alloc::string::String} "logic"

        // Verify tokens are not empty
        assert!(!tokens.is_empty(), "Token list should not be empty");

        // Verify stop words removed
        assert!(
            !tokens.contains(&"to".to_string()),
            "Stop word 'to' should be removed"
        );
        assert!(
            !tokens.contains(&"the".to_string()),
            "Stop word 'the' should be removed"
        );
        assert!(
            !tokens.contains(&"in".to_string()),
            "Stop word 'in' should be removed"
        );

        // Verify stemming applied
        assert!(
            tokens.iter().any(|t| t.starts_with("comput")),
            "Should contain stemmed form of 'computing'"
        );
        assert!(
            tokens.iter().any(|t| t.starts_with("theoret")),
            "Should contain stemmed form of 'theoretical'"
        );
    }

    #[test]
    fn test_nlp_tokenizer_empty_input() {
        let tokenizer = SampleNlpTokenizer::new();
        let tokens = tokenizer.tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_nlp_tokenizer_unicode() {
        let tokenizer = SampleNlpTokenizer::new();
        let tokens = tokenizer.tokenize("café résumé naïve");
        // Tokens:
        // [0] = {alloc::string::String} "cafe"
        // [1] = {alloc::string::String} "resum"
        // [2] = {alloc::string::String} "naiv"

        // Should handle Unicode normalisation
        assert!(tokens.len() == 3);
        assert!(tokens.contains(&"cafe".to_string()));
        assert!(tokens.contains(&"resum".to_string()));
        assert!(tokens.contains(&"naiv".to_string()));
    }
}
