# Learning Graph Neural Networks

A broad learning plan for Graph Neural Networks (GNNs): concepts,
architectures, task types, implementation frameworks, applications, and
production concerns.

PyTorch Geometric (PyG) is only one topic in this plan. Treat it as one
implementation framework among several, not as the whole subject. This note is
meant to be framework-agnostic first, then practical. See `gnn.md` for the
finance-focused GNN + LLM reference, and `CLAUDE.md` for working inside this
specific PyG repository.

---

## 1. Why Graphs

Standard ML treats data as independent rows. Graphs are for problems where *relationships are part of the signal*.

| Domain | Nodes | Edges | What the graph reveals |
|---|---|---|---|
| Financial / AML | Companies, persons, accounts | Owns, transacts, directs | Circular ownership, layering, shell networks |
| Knowledge graphs | Entities, concepts | Typed relations | Missing facts, contradictions, inference |
| Social networks | Users | Follows, messages | Communities, influence, anomalous accounts |
| Molecules | Atoms | Chemical bonds | Molecular properties, drug interactions |
| Cybersecurity | Devices, IPs | Network flows | Lateral movement, unusual patterns |
| Supply chain | Suppliers, factories, ports | Shipments, contracts | Disruption propagation, bottlenecks |

---

## 2. Core Concepts

### Graph Types

**Homogeneous** — one node type, one edge type. Simplest case; loses relation semantics.

**Heterogeneous** — multiple node types and edge types. Required for knowledge graphs and financial data where `:owns` and `:directorOf` mean different things.

**Dynamic / temporal** — edges and nodes change over time. Used for transaction monitoring.

**Directed vs. undirected** — ownership and transactions are directed; social connections are often undirected.

### Message Passing

The core computation in most GNN architectures. Each node iteratively aggregates information from its neighbors:

1. **Message** — each neighbor sends a message derived from its features (and optionally the edge)
2. **Aggregate** — incoming messages are combined via a permutation-invariant function: sum, mean, max, or attention
3. **Update** — the node merges its current state with the aggregated message through a learnable function

After `k` rounds, each node's embedding encodes its `k`-hop neighborhood structure. For AML detection, `k = 2` or `k = 3` lets the model see the full layering pattern around a suspicious account.

### Graph Representation

Graphs are typically stored as:
- **Adjacency matrix** — `[N, N]` dense or sparse matrix; practical only for small graphs
- **Edge list** — list of `(source, destination)` pairs; the universal sparse format
- **CSR / CSC** — compressed sparse row/column; efficient for large-scale operations
- **Framework tensors** — e.g. PyG's `edge_index` (`[2, num_edges]`) or DGL's `DGLGraph` — each library has its own in-memory representation of the edge list

---

## 3. GNN Architectures

### Foundational Architectures

| Architecture | Core idea | Strength | Limitation |
|---|---|---|---|
| **GCN** (Kipf & Welling, 2017) | Spectral convolution approximated as symmetric neighbor averaging | Simple, well-understood baseline | Assumes undirected, homogeneous graph |
| **GraphSAGE** (Hamilton et al., 2017) | Sample and aggregate from fixed-size neighborhood | Scales to large graphs via mini-batching | Fixed sample size loses full neighborhood |
| **GAT** (Veličković et al., 2018) | Attention weights over neighbors | Learns which neighbors matter more | Quadratic attention within neighborhood |
| **GIN** (Xu et al., 2019) | Sum aggregation with MLP; maximally expressive | Theoretically as powerful as 1-WL test | No edge features by default |
| **MPNN** (Gilmer et al., 2017) | General message passing framework | Unifies GCN, GAT, GIN under one abstraction | Framework, not a specific model |

### Heterogeneous Graph Architectures

Required when node types and edge types carry different semantics — the typical case for financial KGs.

| Architecture | Core idea | Best for |
|---|---|---|
| **RGCN** (Schlichtkrüll et al., 2018) | Separate linear transformation per relation type | KGs with many relation types (ownership, directorship, transactions) |
| **HAN** (Wang et al., 2019) | Attention over meta-paths in heterogeneous graphs | When domain meta-paths are known in advance |
| **HGT** (Hu et al., 2020) | Transformer attention across node/edge types | Complex heterogeneous graphs with many types |
| **ieHGCN** | Type-aware interaction + ego network | Fine-grained heterogeneous interaction modeling |

### Knowledge Graph Embedding (distinct from GNNs)

KG embedding models learn entity and relation representations from triples `(head, relation, tail)`.
They are self-supervised and do not require node features — useful when you only have relational data.

| Model | Geometry | Strength |
|---|---|---|
| **TransE** | Translational | Simple; struggles with 1-to-N and symmetric relations |
| **RotatE** | Rotation in complex space | Handles antisymmetric, symmetric, inverse relations |
| **ComplEx** | Complex-valued decomposition | Strong on asymmetric and antisymmetric relations |
| **DistMult** | Bilinear scoring | Fast; only handles symmetric relations |
| **KGCN** | GNN over KG | Combines structural aggregation with KG semantics |

**KG embedding vs. GNN — when to use which:**

| | KG Embedding | GNN |
|---|---|---|
| Input | Typed triples only | Node/edge feature tensors |
| Learns | Geometric relation patterns | Neighborhood structural patterns |
| Best for | Link prediction, UBO inference, entity resolution | Node classification, anomaly detection, contagion |
| Labels required | No (self-supervised) | Yes for supervised tasks |

---

## 4. Key Task Types

### Node Classification / Scoring

Assign a label or score to each node. In fraud detection: score each company or account for anomaly likelihood based on structural patterns, not just attributes.

### Link Prediction

Predict whether an edge exists between two nodes. The mechanism behind:
- UBO inference (predict missing `:beneficialOwnerOf` links)
- Knowledge base completion
- Recommender systems (predict user–item interactions)

Standard approach: GNN encoder produces node embeddings → decoder scores candidate pairs (dot product, MLP, or DistMult) → trained on known positive + sampled negative edges.

### Graph Classification

Assign a label to an entire graph. Used in molecule property prediction (is this compound toxic?), document classification, and anomaly detection at the network level.

### Anomaly / Outlier Detection

Two approaches:
- **Supervised** — train with fraud/non-fraud labels; models like CARE-GNN address camouflaged fraudsters
- **Unsupervised** — embed entities, flag those whose embeddings are outliers; no labels needed, useful when ground truth is scarce

### Graph Generation

Generate new graphs with target properties. Used in drug discovery (generate valid molecules with desired properties) and network synthesis.

---

## 5. Implementation Frameworks

This section is only one part of the learning plan. Learn the graph concepts,
model families, and task formulations before tying your understanding to any
single library.

The main choices are:

- **PyTorch Geometric (PyG):** PyTorch-native, research-friendly, broad model
  coverage.
- **Deep Graph Library (DGL):** Strong large-scale and heterogeneous graph
  support across multiple backends.
- **GraphVite:** Embedding engine for KG and node embeddings, not a general
  GNN framework.
- **Spektral:** Keras / TensorFlow option.
- **PyG Temporal and related tools:** Dynamic and spatiotemporal graph
  learning.

### PyTorch Geometric (PyG)

PyG is the dominant PyTorch-native GNN library and a good practical framework
to learn after the fundamentals. In this broader roadmap, PyG is one
implementation track, especially useful if you already work in PyTorch.

**Key concepts:**
- `Data` — holds a homogeneous graph: `x` (node features), `edge_index`, `edge_attr`, `y`
- `HeteroData` — holds a heterogeneous graph indexed by node type and `(src, relation, dst)` edge type
- `MessagePassing` — base class for all conv layers; implement `message()`, `aggregate()`, `update()`
- `Dataset` / `InMemoryDataset` — base classes for 100+ built-in datasets
- `NeighborLoader` — mini-batch training via k-hop neighborhood sampling

```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

edge_index = torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
x = torch.randn(4, 16)
data = Data(x=x, edge_index=edge_index)

conv = GCNConv(in_channels=16, out_channels=32)
out = conv(data.x, data.edge_index)  # [4, 32]
```

**Heterogeneous graphs in PyG:**

```python
from torch_geometric.data import HeteroData

data = HeteroData()
data['company'].x = torch.randn(100, 32)
data['person'].x = torch.randn(50, 16)
data['company', 'owns', 'company'].edge_index = ...
data['person', 'directorOf', 'company'].edge_index = ...
```

**Custom layer:**

```python
from torch_geometric.nn import MessagePassing

class MyConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j  # features of source node j for each edge
```

**Repo:** [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)

---

### Deep Graph Library (DGL)

Framework-agnostic (PyTorch, TF, MXNet). Strong support for large-scale graphs and heterogeneous data. Favored in enterprise and cloud deployments (AWS, Azure).

**Key concepts:**
- `DGLGraph` — the core graph object
- `apply_edges` / `update_all` — explicit message passing primitives
- `NodeDataLoader` — mini-batch training
- Native heterogeneous graph support via `DGLHeteroGraph`

```python
import dgl
import torch

g = dgl.graph(([0, 1, 1, 2], [1, 2, 3, 3]))
g.ndata['feat'] = torch.randn(4, 16)

from dgl.nn import GraphConv
conv = GraphConv(16, 32)
out = conv(g, g.ndata['feat'])  # [4, 32]
```

**Repo:** [dmlc/dgl](https://github.com/dmlc/dgl)

---

### GraphVite

GPU-accelerated graph embedding engine. Not a GNN framework — it trains KG embedding models (RotatE, TransE, ComplEx) and node embedding models (node2vec, LINE, DeepWalk) at high speed.

Use it when you have relational triples and want self-supervised embeddings without node features.

**Supported models:** TransE, DistMult, ComplEx, RotatE, SimplE, DeepWalk, LINE, node2vec, LargeVis

**Repo:** [DeepGraphLearning/graphvite](https://github.com/DeepGraphLearning/graphvite)

---

### Spektral

Keras / TensorFlow 2 graph learning library. Useful if your team is already in the TF ecosystem.

**Repo:** [danielegrattarola/spektral](https://github.com/danielegrattarola/spektral)

---

### PyG Temporal

Extension of PyG for spatiotemporal and dynamic graphs. Handles graphs that evolve over time — relevant for transaction monitoring.

**Repo:** [benedekrozemberczki/pytorch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)

---

### Framework Comparison

| | PyG | DGL | GraphVite | Spektral |
|---|---|---|---|---|
| Backend | PyTorch | PyTorch / TF / MXNet | PyTorch | Keras / TF2 |
| Heterogeneous graphs | Yes | Yes | No | Limited |
| KG embedding | Via RGCN | Via RGCN | RotatE, TransE, etc. | No |
| Best scale | Research to production | Large-scale / enterprise | Embedding-only | Small to medium |
| Dynamic graphs | Via PyG Temporal | Yes | No | No |

---

## 6. GNN + LLM Integration

Modern systems combine GNNs with LLMs for grounded, explainable question answering over graphs.

**The core pattern:**
```
User query
    ↓ LLM: parse intent, plan retrieval
    ↓ KG: retrieve relevant subgraph
    ↓ GNN: score/embed entities, predict missing links
    ↓ LLM: generate grounded response
    ↓ Agent: iterate if needed
```

**Why combine them:**
- GNN provides structural signal that LLMs cannot derive from text alone
- LLM provides natural language interface and reasoning over retrieved graph context
- Graph structure reduces hallucination by anchoring generation to known relationships

**Key approaches:**

| Approach | How it works | Example |
|---|---|---|
| Graph RAG | LLM extracts KG from text; retrieval via graph traversal | microsoft/graphrag, LightRAG |
| GNN-RAG | GNN retrieves multi-hop reasoning paths; LLM reasons over them | GNN-RAG (arxiv 2405.20139) |
| GNN encoder + LLM | GNN embeds subgraph → embedding injected into LLM context | LLaGA, HiGPT, GOFA |
| LLM as KG builder | LLM extracts triples from text to populate the KG | text2graph_llm, FinDKG |

See `gnn.md` Part 5 for specific repos and `gnn.md` Part 4 for full financial system designs.

---

## 7. Application Domains

GNNs are useful across many domains. Financial AI is one important application,
but it is not the only reason to learn GNNs.

| Domain | Common graph | Typical tasks | Useful architectures |
|---|---|---|---|
| Knowledge graphs | Entities and typed relations | Link prediction, entity resolution, consistency checking | RGCN, HGT, KG embeddings |
| Finance / AML | Companies, people, accounts, transactions | Fraud scoring, UBO inference, contagion analysis | RGCN, HGT, GraphSAGE, RotatE |
| Recommender systems | Users, items, interactions | Recommendation, ranking, cold-start retrieval | GraphSAGE, GAT, LightGCN |
| Molecules and biology | Atoms, bonds, proteins, interactions | Property prediction, drug discovery, interaction prediction | MPNN, GIN, GCN |
| Cybersecurity | Devices, IPs, users, flows | Intrusion detection, lateral movement detection, anomaly scoring | GAT, GraphSAGE, RGCN |
| Social networks | Users and interactions | Community detection, influence prediction, bot detection | GCN, GAT, GraphSAGE |
| Supply chain | Suppliers, factories, ports, shipments | Disruption propagation, bottleneck detection, optimization | GraphSAGE, temporal GNNs |

### Financial Examples

See `gnn.md` Part 4 for full system designs covering KYC/AML, UBO inference, credit risk contagion, regulatory Q&A, and investment research — each with problem statement, system flow, architecture choice, and recommended repos.

---

## 8. Learning Path

### Stage 1 — Graph Foundations

1. Graph terminology: nodes, edges, directed/undirected, homogeneous/heterogeneous, adjacency matrix
2. Graph data representations: adjacency matrix, edge list, sparse tensors, CSR/CSC
3. Core graph algorithms: BFS/DFS, shortest paths, connected components, PageRank, community detection

### Stage 2 — GNN Concepts

4. Message passing: why repeated neighborhood aggregation creates useful embeddings
5. Expressiveness and limits: over-smoothing, over-squashing, graph isomorphism, 1-WL intuition
6. Key architectures: GCN, GraphSAGE, GAT, GIN, RGCN, HAN, HGT

### Stage 3 — Core Tasks

7. Node classification and node scoring
8. Link prediction and knowledge graph completion
9. Graph classification and molecular property prediction
10. Anomaly detection and unsupervised embedding analysis

### Stage 4 — Framework Practice

11. Pick one general GNN framework: PyG or DGL
12. Run a benchmark node-classification example, such as Cora or OGBN-Arxiv
13. Implement a small GCN, then swap in GraphSAGE and GAT
14. Try one heterogeneous graph example with RGCN or HGT
15. Separately, train a KG embedding model such as RotatE or ComplEx with GraphVite or another KG toolkit

**If you chose PyG**, the key primitives to learn at this stage: `Data`, `HeteroData`, `edge_index`, `MessagePassing`, built-in layers (`GCNConv`, `SAGEConv`, `GATConv`, `RGCNConv`), `NeighborLoader`, and dataset/transform patterns. See `CLAUDE.md` for the repo setup.

### Stage 5 — Applied Projects

16. Build one project in a non-financial domain, such as molecule classification or recommendation
17. Build one knowledge graph project with link prediction
18. Build one financial graph project, such as AML anomaly scoring or UBO inference
19. Add an LLM or Graph RAG layer only after the graph model and retrieval logic work independently

### Stage 6 — Production and Evaluation

20. Scale with mini-batch training (`NeighborLoader` in PyG, `NodeDataLoader` in DGL)
21. Evaluate with task-appropriate metrics: accuracy/F1 for node classification, precision@k/MRR for link prediction, AUROC/AUPRC for anomaly detection
22. Track graph leakage, temporal splits, negative sampling strategy, and class imbalance
23. Plan deployment: feature pipelines, graph refresh cadence, model monitoring, explainability, and human review workflows
