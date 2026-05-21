# Learning Graph Neural Networks

> **Focus:** GNNs for financial use cases — ownership intelligence, AML/KYC, credit risk. Implementation in PyTorch Geometric (PyG).
> See `gnn.md` for the full tool reference and application use case designs.

---

## 1. Why Graphs for Financial Data

Financial data is inherently relational. The signal isn't just in entity attributes — it's in the *structure* of connections between them.

| Problem | Why tabular ML fails | Why graphs work |
|---|---|---|
| Detecting circular ownership | Can't see multi-hop loops | Graph traversal finds cycles directly |
| UBO inference | Missing rows can't be predicted | Link prediction over structural patterns |
| AML layering detection | Transactions look normal individually | Structural fingerprint across the chain |
| Counterparty contagion | Exposure is hidden behind intermediaries | Graph propagation traces indirect paths |
| Peer group discovery | SIC codes are too coarse | Embedding clusters by relational structure |

---

## 2. Core Concepts

### Graphs

A graph `G = (V, E)` consists of nodes `V` (entities: companies, persons, accounts, jurisdictions) and edges `E` (relationships: `:owns`, `:transactedWith`, `:directorOf`).

**Homogeneous graph** — one node type, one edge type. Simple, but loses relation semantics.

**Heterogeneous graph** — multiple node types and edge types. Essential for financial KGs where the type of relationship (`:owns` vs. `:directorOf`) carries meaning.

**Key formats in PyG:**
- `edge_index`: a `[2, num_edges]` tensor of (source, target) pairs — the standard sparse format
- `Data`: holds a homogeneous graph (`x`, `edge_index`, `edge_attr`, `y`)
- `HeteroData`: holds a heterogeneous graph, indexed by node type and edge type tuple

### Message Passing

The core operation in most GNNs. Each node repeatedly aggregates information from its neighbors:

1. **Message** — each neighbor computes a message based on its features and the edge
2. **Aggregate** — messages from all neighbors are combined (sum, mean, max, attention)
3. **Update** — the node updates its embedding using its current state + aggregated messages

After `k` rounds, each node's embedding encodes the structure of its `k`-hop neighborhood. For AML, `k=2` or `k=3` lets the model see the full layering pattern around a suspicious account.

### Minimal PyG example

```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# A small ownership graph: 4 companies, 4 directed edges
edge_index = torch.tensor([
    [0, 1, 1, 2],   # source nodes
    [1, 2, 3, 3],   # target nodes
], dtype=torch.long)

x = torch.randn(4, 16)          # 4 nodes, 16 features each
data = Data(x=x, edge_index=edge_index)

conv = GCNConv(in_channels=16, out_channels=32)
out = conv(data.x, data.edge_index)  # [4, 32] — one embedding per node
```

---

## 3. Architecture Decision Guide

### KG Embedding vs. GNN — When to Use Which

| | KG Embedding (RotatE, TransE) | GNN (GCN, GAT, RGCN) |
|---|---|---|
| **Input** | (head, relation, tail) triples | Node/edge feature tensors |
| **Learns** | Geometric relation patterns in embedding space | Structural neighborhood patterns |
| **Best for** | Link prediction, entity resolution, UBO inference | Node classification, anomaly detection, contagion |
| **Tool** | GraphVite | PyTorch Geometric |
| **Requires labels?** | No (self-supervised) | Yes for supervised tasks |

**Rule of thumb:** Use KG embedding when you have relation-typed triples and want to predict missing links. Use GNN when you have node/edge features and want to classify or score entities.

### Choosing a GNN Architecture

| Architecture | Mechanism | Best for in financial graphs |
|---|---|---|
| GCN | Symmetric aggregation over neighbors | Baseline node embeddings, contagion modeling |
| GAT | Attention-weighted neighbor aggregation | Heterogeneous importance (some edges matter more) |
| GraphSAGE | Sampled neighborhood aggregation | Large graphs where full-batch is infeasible |
| RGCN | Separate weights per relation type | Heterogeneous KGs — different relation types need different treatment |
| HGT | Transformer attention over heterogeneous graphs | Complex KGs with many node/edge types |

**For financial KGs:** RGCN or HGT because the graph is heterogeneous (companies, persons, jurisdictions, instruments, transactions all have different semantics).

---

## 4. Heterogeneous Graphs in PyG

Financial RDF data maps naturally to `HeteroData`:

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Node features by type
data['company'].x = torch.randn(100, 32)      # 100 companies
data['person'].x = torch.randn(50, 16)        # 50 persons
data['jurisdiction'].x = torch.randn(20, 8)   # 20 jurisdictions

# Edges by type — tuple: (src_type, relation, dst_type)
data['company', 'owns', 'company'].edge_index = ...
data['person', 'directorOf', 'company'].edge_index = ...
data['company', 'registeredIn', 'jurisdiction'].edge_index = ...
```

Use `RGCNConv` or `HGTConv` to process this graph — they handle multiple relation types natively.

---

## 5. Key Task Types

### Node Classification / Scoring

Label nodes (companies, accounts) as suspicious or clean. The GNN learns which structural patterns — not just individual attributes — correlate with fraud.

```python
from torch_geometric.nn import RGCNConv

conv = RGCNConv(in_channels=32, out_channels=64, num_relations=5)
# num_relations = number of distinct edge types in your KG
```

### Link Prediction

Predict missing edges — the mechanism behind UBO inference. Train on known ownership links, then score unobserved `(entity_A, :beneficialOwnerOf, entity_B)` pairs.

Standard approach: GNN encoder → dot product decoder → binary cross-entropy on positive + negative edge pairs.

### Anomaly Detection

Two approaches:
1. **Supervised** — train with fraud/non-fraud labels on nodes or edges (CARE-GNN)
2. **Unsupervised** — train embeddings, flag entities whose embeddings are outliers in the embedding space (no labels needed; works well when ground truth is scarce)

---

## 6. Working with PyG

### Data loading pattern

```python
from torch_geometric.datasets import Planetoid        # for prototyping
from torch_geometric.loader import NeighborLoader     # for large graphs

# NeighborLoader samples k-hop neighborhoods for mini-batch training
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],   # sample 25 neighbors at hop 1, 10 at hop 2
    batch_size=64,
    input_nodes=train_mask,
)
```

### Custom MessagePassing layer

All conv layers in PyG subclass `MessagePassing`. To implement a custom layer:

```python
from torch_geometric.nn import MessagePassing

class MyConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')   # or 'sum', 'max', or an Aggregation module

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j   # x_j: features of source node j for each edge
```

`propagate()` calls `message()` → aggregates → calls `update()`. Override only what you need.

### Testing a model

```bash
# Run a specific test
pytest test/nn/conv/test_gcn_conv.py

# Run with FULL_TEST for slow/extended tests
FULL_TEST=1 pytest test/nn/conv/
```

---

## 7. Financial Use Case Map

| Use Case | GNN task | Architecture | Layer in the stack |
|---|---|---|---|
| AML / suspicious entity flagging | Node scoring (anomaly detection) | RGCN or HGT | GNN inference |
| UBO / beneficial ownership inference | Link prediction | RotatE (GraphVite) or RGCN | KG completion |
| Credit risk contagion | Node embedding + clustering | node2vec or GraphSAGE | GNN embedding |
| Regulatory Q&A | Multi-hop graph traversal | RGCN | KG + LLM retrieval |
| Peer group discovery | Unsupervised clustering | LINE or node2vec | GNN embedding |

See `gnn.md` Part 4 for the full system designs: data flow, LLM integration, agent frameworks, and recommended repos for each.

---

## 8. Learning Path

1. **Graph basics** — nodes, edges, directed vs. undirected, homogeneous vs. heterogeneous, adjacency matrix, `edge_index` format
2. **Message passing** — implement the minimal `MessagePassing` example above; trace what `propagate()` does
3. **Node classification on Cora** — use `Planetoid` dataset, train a 2-layer GCN, understand train/val/test masks
4. **Heterogeneous graphs** — convert a small ownership triple dataset to `HeteroData`, run `RGCNConv`
5. **Link prediction** — implement encoder + decoder, train on positive/negative edge pairs
6. **Scale up** — replace full-batch with `NeighborLoader` for mini-batch training
7. **Financial domain** — wire a real or synthetic ownership KG through the KYC/AML pipeline in `gnn.md`
