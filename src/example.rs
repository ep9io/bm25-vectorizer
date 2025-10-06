/// Example demonstrating a Murmur3-based token indexer and a lightweight NLP tokenizer.
///
/// Dependencies required for the indexer example:
/// - `hash32` 1.0.0
///
/// Dependencies required for the tokenizer example:
/// - `unicode-segmentation` 1.12.0
/// - `stop-words` 0.9.0 (with `nltk` feature)
/// - `rust-stemmers` 1.2.0
/// - `deunicode` 1.6.2
/// - `regex` 1.11.3
///
#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use crate::{Bm25TokenIndexer, SparseRepresentation, TokenIndexValue};
    use crate::{Bm25Tokenizer, Bm25VectorizerBuilder};
    use hash32::Murmur3Hasher;
    use regex::Regex;
    use rust_stemmers::{Algorithm as StemmingAlgorithm, Stemmer};
    use std::hash::Hasher;
    use stop_words::{get, LANGUAGE as StopWordLanguage};
    use unicode_segmentation::UnicodeSegmentation;
    use lazy_static::lazy_static;

    lazy_static! {
    static ref NON_ALPHANUM_REGEX: Regex = Regex::new(r"[^\w\s]").unwrap();
    static ref ENGLISH_STOP_WORDS: HashSet<String> = {
        get(StopWordLanguage::English)
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
}


    pub struct SimpleEnglishTokenizer {
        stemmer: Stemmer,
        stop_words: HashSet<String>,
    }

    impl SimpleEnglishTokenizer {
        pub(crate) fn new() -> Self {
            Self {
                stemmer: Stemmer::create(StemmingAlgorithm::English),
                stop_words: {
                    get(StopWordLanguage::English)
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                }
            }
        }
    }

    impl Bm25Tokenizer for SimpleEnglishTokenizer {
        /// Tokenizes input text through a NLP pipeline.
        ///
        /// # Processing Steps
        ///
        /// 1. Unicode normalisation.  Non‑ASCII characters are replaced with ASCII equivalents.
        /// 2. All non‑alphanumeric symbols are removed.
        /// 3. Text is converted to lowercase.
        /// 4. Word segmentation.  Unicode word boundaries are respected.
        /// 5. Stop‑word filtering. Common English words are discarded (e.g., "the", "is", "at").
        /// 6. Stemming. Tokens are reduced to their stem form (e.g., "running" → "run").
        ///
        fn tokenize(&self, input_text: &str) -> Vec<String> {
            // Step 1: Normalise Unicode characters to ASCII
            // U+FFFD � REPLACEMENT CHARACTER used to replace an unknown, unrecognised, or unrepresentable character
            let text = deunicode::deunicode_with_tofu_cow(input_text, "�");

            // Step 2: Remove non-alphanumeric characters
            let text = NON_ALPHANUM_REGEX.replace_all(&text, " ");

            // Step 3: Convert to lowercase
            let text = text.to_lowercase();

            // Steps 4, 5 & 6: Tokenise, filter stop words, and stem in a single pass
            text.unicode_words()
                .filter(|word| !word.is_empty() && !ENGLISH_STOP_WORDS.contains(*word))
                .map(|token| self.stemmer.stem(token).to_string())
                .collect()
        }
    }

    struct Murmur3Indexer;

    impl Bm25TokenIndexer for Murmur3Indexer {
        type Bm25TokenIndex = u32;

        fn index(&self, token: &str) -> Self::Bm25TokenIndex {
            let mut hasher: Murmur3Hasher = Default::default();
            hasher.write(token.as_bytes());
            hasher.finish() as u32
        }
    }

    #[test]
    fn example() {
        let corpus = [
            "Alan Turing’s pioneering work laid the conceptual groundwork for modern artificial \
            intelligence by formalizing the notion of computation and demonstrating that machines \
            could, in principle, simulate any algorithmic process. In the early 1930s, he \
            introduced the abstract machine now known as the Turing machine, a minimal yet \
            universal computational device that could, given an appropriate program, perform any \
            calculation that a human could. By 1936, his seminal paper “On Computable Numbers” \
            established the undecidability of the Halting Problem, revealing inherent limits on \
            algorithmic reasoning. Decades later, during the 1950s, Turing applied these insights \
            to the nascent field of machine learning, proposing that an intelligent system could \
            be evaluated by its ability to imitate human responses—an idea later formalized as the \
            Turing Test. His 1950 essay “Computing Machinery and Intelligence” critically examined \
            the philosophical question of machine minds, arguing that, if a machine could not be \
            distinguished from a human in a conversational setting, it should be considered \
            intelligent. Turing also anticipated concepts central to neural network research, such \
            as pattern recognition and the need for adaptive learning mechanisms, through his work \
            on the Turing Test and subsequent essays on the possibilities of symbolic and \
            sub-symbolic representation. Collectively, these contributions constitute the first \
            substantial, coherent research agenda for artificial intelligence, framing it as a \
            discipline grounded in formal mathematics, computability theory, and empirical \
            experimentation.",
        ];

        let vectorizer = Bm25VectorizerBuilder::new()
            .tokenizer(SimpleEnglishTokenizer::new())
            .token_indexer(Murmur3Indexer)
            .avgdl(256.0)
            .k1(1.2)
            .b(0.75)
            .build()
            .expect("Could not build vectorizer");

        assert_eq!(vectorizer.avgdl(), 256.0);
        assert_eq!(vectorizer.b(), 0.75);
        assert_eq!(vectorizer.k1(), 1.20000005);
        assert_eq!(vectorizer.delta(), 0.0);

        let embedding = vectorizer.vectorize(corpus[0]);

        assert_eq!(
            embedding,
            SparseRepresentation(vec![
                TokenIndexValue {
                    index: 14662688,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 34379837,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 129810225,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 143757772,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 150695120,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 189262593,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 301705055,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 367094702,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 412017637,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 520409122,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 540174517,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 588610688,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 609543353,
                    value: 1.8968072
                },
                TokenIndexValue {
                    index: 614556384,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 614665414,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 644716969,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 695743380,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 742500863,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 804460016,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 818409348,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 900034473,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 930883039,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 964745590,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1014127006,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1050980247,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1082468256,
                    value: 1.7371994
                },
                TokenIndexValue {
                    index: 1098657283,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1190196098,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1200518484,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 1230423685,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1290213554,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1308688855,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1330437081,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1376864891,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1428905997,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 1491351846,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1508140589,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1545193637,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1590456296,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1620377129,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1704236722,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1706587157,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1731862276,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1758470515,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1856538418,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1920135276,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1943344364,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 1997889251,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2002242733,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2030235830,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2032101475,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2095749492,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2137471997,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2179962017,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2194093370,
                    value: 1.8968072
                },
                TokenIndexValue {
                    index: 2198099337,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2206024373,
                    value: 1.8968072
                },
                TokenIndexValue {
                    index: 2210611672,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2228746633,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2231220596,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2373257806,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2558937123,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2570541023,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2622103353,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2650797237,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 2851137560,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2854705516,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2876684594,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2882792817,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2886837394,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 2954664702,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 3066577729,
                    value: 1.9413997
                },
                TokenIndexValue {
                    index: 3087468851,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3113539432,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3127628307,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 3168814557,
                    value: 1.7371994
                },
                TokenIndexValue {
                    index: 3169584877,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3177574277,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3250766381,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3259053156,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3284920822,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3316981470,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3335033140,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3340191823,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3411418812,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3525523449,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 3593645578,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3639641146,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3660156378,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3719599989,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3741620795,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3812103426,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3854329402,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3874492524,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3876873682,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3878249820,
                    value: 1.9413997
                },
                TokenIndexValue {
                    index: 3927490055,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3940191930,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3972741056,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 3997133275,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 4006988776,
                    value: 1.5718672
                },
                TokenIndexValue {
                    index: 4124048943,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 4183835765,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 4238889041,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 4257735560,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 4278052117,
                    value: 1.2227529
                },
                TokenIndexValue {
                    index: 4279915734,
                    value: 1.2227529
                },
            ])
        )
    }
}
