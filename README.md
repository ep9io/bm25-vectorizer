# BM25 Vectorizer

A minimal Rust library for creating sparse vector representations (embeddings) using the BM25 algorithm. These embeddings can be loaded onto vector databases like Qdrant, Milvus, and others for information retrieval.

## Why this library? 

This library was created to address the following gaps with existing solutions: 

- A minimal‑dependency library for generating BM25 embeddings that can be loaded into vector databases.  Only `thiserror` is required (the `rayon` crate is optional for parallelism).
- Separation of concerns. Tokenisation and indexing are decoupled, allowing the user to choose hashing strategies (e.g. Murmur3, dictionary, etc.) or custom tokenisers.
- No duplicate indices/values. The final embedding vector contains unique indices.
- Reproducible indices/values. Avoids HashMap to guarantee deterministic results (e.g. downstream unit tests).
     

## Features 

- Minimal dependencies.
- Flexible tokenisation & indexing.  Use your own tokeniser or indexing scheme (hashing/dictionary).
- Parallel processing.  Optional support via the `rayon` crate.