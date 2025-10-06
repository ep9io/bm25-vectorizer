# BM25 Vectorizer

A minimal Rust library for creating sparse vector representations using the BM25 algorithm. These vectors can be efficiently stored in vector databases like Qdrant for keyword-based information retrieval.

BM25 is a probabilistic ranking algorithm that calculates relevance scores between queries and documents based on term frequency and inverse document frequency. This library's implementation produces only the normalised term frequency (TF) component in document vectors and expects the inverse document frequency (IDF) to be computed by the vector database. This approach allows IDF to automatically update as documents are added or removed without re-encoding existing documents.

NOTE: Vector databases might require to specify an IDF modifier when setting up the vector store to instruct them to calculate IDF statistics automatically.

## Why this library? 

This library was created to address the following gaps with existing Rust solutions (Sep 2025): 

- A minimal‑dependency library for generating BM25 embeddings that can be loaded onto vector databases.  Only `thiserror` crate is required (the `rayon` crate is optional for parallelism).
- Separation of concerns. Tokenisation and indexing are decoupled, allowing the dependent library/binary to choose hashing (e.g. Murmur3, dictionary, etc.) and tokeniser strategies.
- No duplicate indices/values. The final embedding vector contains unique indices.
- Support for BM25+ delta (δ) parameter to ensure minimum contribution from matching terms.
- Reproducible indices/values. This implementation avoids HashMap to guarantee deterministic results (e.g. downstream unit tests).
     

## Features 

- Minimal dependencies.
- Flexible tokenisation & indexing.  Use your own tokeniser or indexing scheme (hashing/dictionary).
- Optional BM25+ support: adds delta (δ) to ensure minimum contribution from matching terms.
- Parallel processing.  Optional support via the `rayon` crate.

# Usage

The file [example.rs](src/example.rs) provides an example of implementing a Murmur3 indexer and a tokenizer that performs the following steps:
1. Unicode normalisation.  Non‑ASCII characters are replaced with ASCII equivalents.
2. All non‑alphanumeric symbols are removed.
3. Text is converted to lowercase.
4. Word segmentation.  Unicode word boundaries are respected.
5. Stop‑word filtering. Common English words are discarded (e.g., "the", "is", "at").
6. Stemming. Tokens are reduced to their stem form (e.g., "running" → "run").