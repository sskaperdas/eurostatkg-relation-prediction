This repository provides a modular and ontology-aware framework for relation prediction in schema-rich knowledge graphs, using the EurostatKG as a case study. The system reformulates relation prediction as a triple classification task, where the goal is to determine if a given RDF triple is valid—both factually and semantically. It combines classical embedding models (like TransE, DistMult, and ComplEx) with graph-based encoders (GCN and GAT), and integrates symbolic constraints such as rdf:type, rdfs:domain, and rdfs:range directly into the training loop. The /models directory contains multiple trained models and experiment configurations, including ontology-guided variants that differ in architecture, loss function, and classifier design. This setup supports both standard ranking-based evaluation and more nuanced classification using hybrid losses and type-aware negative sampling. The project offers a reproducible pipeline for experimenting with neural-symbolic KG completion methods in domains where ontological structure matters.
The /models directory contains multiple subfolders named GOE_1, GOE_4, ..., each corresponding to a different model configuration evaluated in the ontology-guided relation prediction framework. These experiments vary in terms of embedding strategy, graph encoder, classifier head, and whether ontology constraints are integrated. Specifically:

GOE_1 uses MLP + embeddings (baseline).

GOE_4 adds a GCN with type embeddings: MLP + GCN + embeddings.

GOE_5 swaps GCN for GAT: MLP + GAT + embeddings.

GOE_6 combines both: MLP + GCN + GAT + embeddings.

GOE_7 switches to a Conv1D classifier + embeddings.

GOE_8 adds GCN: Conv1D + GCN + embeddings.

GOE_9 uses Conv1D + GAT + embeddings.

GOE_10 combines both encoders: Conv1D + GCN + GAT + embeddings.

GOE_11 introduces the full ontology-aware pipeline with MLP + embeddings + constraints.

GOE_14 builds on that with MLP + GCN + embeddings + constraints.

GOE_15 uses MLP + GAT + embeddings + constraints.

GOE_16 combines everything: MLP + GCN + GAT + embeddings + full ontology constraints.
