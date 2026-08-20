Created a complete, portfolio-ready documentation and demonstration suite.
Key deliverables:
- [README.md](/home/pierre/projects/leiden-ai/README.md) — architecture diagrams, configuration tables, quick start, repository layout, limitations, and documentation links.
- [Setup guide](/home/pierre/projects/leiden-ai/docs/setup.md) — installation, dependencies, CPU configuration, resource considerations, and verification.
- Individual algorithm guides:
  - [Sentence embeddings](/home/pierre/projects/leiden-ai/docs/sentence-embeddings.md)
  - [HNSW indexing](/home/pierre/projects/leiden-ai/docs/hnsw.md)
  - [Weighted k-NN graph](/home/pierre/projects/leiden-ai/docs/knn-graph.md)
  - [Hierarchical Leiden](/home/pierre/projects/leiden-ai/docs/hierarchical-leiden.md)
  - [Cluster representatives](/home/pierre/projects/leiden-ai/docs/cluster-representatives.md)
- [Animal pipeline notebook](/home/pierre/projects/leiden-ai/notebooks/animal_hierarchy_pipeline.ipynb)
- [Synthetic animal dataset](/home/pierre/projects/leiden-ai/notebooks/animal_sentences.json) — 96 sentences across 3 habitats, 5 animal classes, and 12 species.
The notebook was executed end to end on CPU using all-MiniLM-L6-v2. All 10 code cells passed, producing:
- 96 × 384 normalized embeddings
- 96 graph vertices and 466 weighted edges
- A hierarchy reaching depth three
- Representative sentences and reference-label purity analysis
I also fixed a missing ClusterNode import in [src/leiden.py (line 28)](/home/pierre/projects/leiden-ai/src/leiden.py:28) discovered during real pipeline execution. Python compilation, JSON validation, dataset validation, and git diff --check all pass. The unrelated existing prompts/ files were left untouched.
