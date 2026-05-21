# GNN + LLM — Financial AI Reference

> **Focus:** Applying knowledge graphs, GNNs, and LLMs to financial use cases — ownership intelligence, fraud/AML detection, credit risk, and regulatory compliance.

## How to Read This Document

| Part | What it covers | Start here if... |
|---|---|---|
| [Part 1](#part-1-financial-knowledge-graph-embedding) | KG embedding models for financial RDF data | You're working with ownership/transaction triples |
| [Part 2](#part-2-open-source-ecosystem) | GNN frameworks, Graph RAG, and LLM infrastructure | You need to choose tools |
| [Part 3](#part-3-open-source-llm-models) | Open source LLM model families | You need to pick a local LLM |
| [Part 4](#part-4-application-use-cases) | Financial use cases — problem, system design, stack | You want to build something |
| [Part 5](#part-5-gnn--llm-hybrid-research) | GNN+LLM hybrid models and graph foundation models | You're tracking research |
| [Part 6](#part-6-curated-collections) | Paper lists and awesome-repos | You want to go deeper |

---

## Part 1: Financial Knowledge Graph Embedding

### Data Model

Financial RDF triples typically look like:

```
:CompanyA  :owns              :CompanyB
:CompanyA  :headquarteredIn   :Jurisdiction_X
:PersonA   :directorOf        :CompanyB
:CompanyA  :counterpartyOf    :CompanyC
:LoanA     :issuedTo          :CompanyA
:LoanA     :securedBy         :AssetX
```

### Embedding Approaches

#### Knowledge Graph Embedding (relation-aware)

Preserves relation types — essential when the *type* of relationship matters (`:ownerOf` is not the same as its inverse).

**Fraud / AML detection**
Train RotatE on ownership + transaction triples. Entities in suspicious structures (circular flows, unusual relation combinations) have anomalous embeddings — flag as outliers or use as features in a classifier.

**UBO / KB completion**
FIBO/LEI data is incomplete. Train on known ownership triples, predict missing `:beneficialOwnerOf` and `:controlledBy` links. Directly actionable for KYC workflows.

**Counterparty / credit risk**
Link prediction over `(:Company :hasExposureTo :Company)`. The model learns that companies sharing relation patterns (same jurisdiction, same creditor) cluster together — surfaces implicit risk propagation.

**Entity resolution across sources**
Merge RDF graphs from Bloomberg, Refinitiv, SEC EDGAR. Find entities whose embeddings are nearest neighbors across source-specific subgraphs.

#### Node Embedding (topology-aware)

Flatten relations, keep entity-entity edges. Use when you care about network structure, not relation types.

**Systemic risk / contagion**
Train node2vec. High-centrality nodes in embedding space = systemically important. Captures second/third-order structural role, not just degree.

**Sector / peer group discovery**
Cluster embeddings on co-investment or co-counterparty graph. Emergent clusters reveal undisclosed sector exposure invisible in SIC/NAICS codes.

**Portfolio structural correlation**
Two assets look uncorrelated by price but share common counterparties or ownership. Use embedding cosine similarity as a structural correlation signal.

#### Visualization

Plot entity embeddings colored by `rdf:type` (`:Company`, `:Person`, `:Jurisdiction`, `:Instrument`). Shell company clusters and hub nodes (nominee directors appearing in many chains) stand out visually. Flag entities that land in the wrong cluster — a `:NaturalPerson` near `:ShellCompany` nodes is likely a nominee director or data error.

### Recommended Model by Task

| Task | Model | Why |
|---|---|---|
| AML / fraud detection | RotatE | Handles antisymmetric relations (`:ownerOf` ≠ reverse) |
| KB completion / UBO inference | RotatE or ComplEx | Best accuracy on asymmetric relations |
| Systemic risk topology | node2vec | BFS/DFS balance captures local and global structure |
| Peer group / sector clustering | LINE | Explicit 2nd-order proximity = shared-neighbor grouping |
| Visualization / audit | LargeVis on KG embeddings | Visualize entity space after RotatE |

---

## Part 2: Open Source Ecosystem

### GNN Frameworks

| Repo | Link | Backend | Key algorithms |
|---|---|---|---|
| PyTorch Geometric (PyG) | [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | PyTorch | GCN, GAT, GraphSAGE, GIN, RGCN, HGT, GraphTransformer |
| Deep Graph Library (DGL) | [dmlc/dgl](https://github.com/dmlc/dgl) | PyTorch / TF / MXNet | GCN, GAT, GraphSAGE, GIN, RGCN, GATv2 |
| GraphVite | [DeepGraphLearning/graphvite](https://github.com/DeepGraphLearning/graphvite) | PyTorch (CPU+GPU) | DeepWalk, LINE, node2vec, TransE, RotatE, LargeVis |
| Spektral | [danielegrattarola/spektral](https://github.com/danielegrattarola/spektral) | Keras / TF 2 | GCN, GAT, GraphSAGE, spectral methods |
| PyG Temporal | [benedekrozemberczki/pytorch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) | PyTorch | Spatiotemporal / dynamic graphs |

### Individual GNN Model Implementations

| Model | Repo | Notes |
|---|---|---|
| GAT | [PetarV-/GAT](https://github.com/PetarV-/GAT) | Official TF implementation |
| GAT (PyTorch) | [gordicaleksa/pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT) | Tutorial-quality |
| GraphSAGE | [williamleif/GraphSAGE](https://github.com/williamleif/GraphSAGE) | Official implementation |
| RGCN | [tkipf/relational-gcn](https://github.com/tkipf/relational-gcn) | Official Keras |
| RGCN (PyTorch) | [thiviyanT/torch-rgcn](https://github.com/thiviyanT/torch-rgcn) | PyTorch port |
| HGT | [acbull/pyHGT](https://github.com/acbull/pyHGT) | Heterogeneous Graph Transformer, PyG |

### Fraud / Anomaly Detection on Graphs

| Repo | Link | Notes |
|---|---|---|
| DGFraud | [safe-graph/DGFraud](https://github.com/safe-graph/DGFraud) | Multi-algorithm deep graph fraud detection toolbox |
| CARE-GNN | [YingtongDou/CARE-GNN](https://github.com/YingtongDou/CARE-GNN) | Against camouflaged fraudsters; relation-aware attention (CIKM 2020) |
| AWS Realtime Fraud | [awslabs/realtime-fraud-detection-with-gnn-on-dgl](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl) | DGL + Neptune + SageMaker end-to-end pipeline |
| Graph Fraud Papers | [safe-graph/graph-fraud-detection-papers](https://github.com/safe-graph/graph-fraud-detection-papers) | Curated paper list |

### Knowledge Graph + LLM

| Repo | Link | Notes |
|---|---|---|
| GNN-RAG | [cmavro/GNN-RAG](https://github.com/cmavro/GNN-RAG) | GNN retrieves reasoning paths; LLM reasons over them; SOTA on KGQA with 7B |
| KG_RAG | [BaranziniLab/KG_RAG](https://github.com/BaranziniLab/KG_RAG) | Task-agnostic KG + LLM |
| GNN-QE | [DeepGraphLearning/GNN-QE](https://github.com/DeepGraphLearning/GNN-QE) | Logical query execution over KGs (ICML 2022) |
| GoG | [YaooXu/GoG](https://github.com/YaooXu/GoG) | LLM as both agent and KG for incomplete KGQA (EMNLP 2024) |
| text2graph_llm | [UW-xDD/text2graph_llm](https://github.com/UW-xDD/text2graph_llm) | LLM extracts (subject, predicate, object) triplets from text |

### Graph RAG

| Repo | Link | Notes |
|---|---|---|
| GraphRAG | [microsoft/graphrag](https://github.com/microsoft/graphrag) | LLM extracts KG from text; community hierarchies; global + local query modes |
| LightRAG | [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | Dual-level (entity + relation) retrieval; graph-vector hybrid; cheaper than GraphRAG |
| nano-graphrag | [gusye1234/nano-graphrag](https://github.com/gusye1234/nano-graphrag) | ~800-line GraphRAG reimplementation; good for understanding internals |
| GeAR | [arxiv:2412.18431](https://arxiv.org/abs/2412.18431) | Graph expansion + multi-step retrieval agent (ACL 2025) |

### RAG & Agent Frameworks

| Project | Repo | Notes |
|---|---|---|
| LangChain | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | Agent orchestration, RAG chains |
| LlamaIndex | [run-llama/llama_index](https://github.com/run-llama/llama_index) | Data framework for LLM apps; strong graph connectors |
| Dify | [langgenius/dify](https://github.com/langgenius/dify) | Visual workflow builder, RAG pipelines |
| RAGFlow | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Enterprise RAG engine + agents |
| CrewAI | [joaomdmoura/crewai](https://github.com/joaomdmoura/crewai) | Multi-agent role-playing |

### LLM Inference

| Framework | Repo | Best for |
|---|---|---|
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | High-throughput production, multi-GPU |
| Ollama | [ollama/ollama](https://github.com/ollama/ollama) | Local prototyping, Docker-style UX |
| llama.cpp | [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | Portable CPU/GPU inference, edge devices |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | Structured generation, RAG |

### LLM Fine-tuning

| Framework | Repo | Key methods |
|---|---|---|
| Axolotl | [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | LoRA, QLoRA, DPO, GRPO, FSDP/DeepSpeed |
| Unsloth | [unslothai/unsloth](https://github.com/unslothai/unsloth) | 2× faster, 70% less VRAM |
| TRL | [huggingface/trl](https://github.com/huggingface/trl) | RLHF, DPO, KTO, ORPO, GRPO |
| LLaMA-Factory | [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 100+ models, Unsloth integration |

---

## Part 3: Open Source LLM Models

| Family | Org | Sizes | License | Highlights |
|---|---|---|---|---|
| Llama 3 / 3.1 / 3.2 / 4 | Meta | 1B–405B (+ MoE) | Apache 2.0 | 128K context (3.1), MoE in Llama 4 |
| Mistral / Mistral Small | Mistral AI | 7B–123B | Apache 2.0 | Strong coding, instruction following |
| Qwen 2 / 3 / 3.5 | Alibaba | 0.5B–235B MoE | Apache 2.0 | Top multilingual, 201 languages |
| Gemma 2 / 3 / 4 | Google | 1B–31B | Apache 2.0 | Memory-efficient; 4B = 4.2 GB RAM |
| Phi-3 / Phi-4 | Microsoft | 3.8B–14B | MIT | Strong reasoning per parameter |
| DeepSeek V3 / R1 | DeepSeek | 7B–671B MoE | MIT | R1 competitive with OpenAI o1 on math/code |

**For local financial AI agents:** Llama 3.1 8B or Qwen 2.5 7B via Ollama are the practical starting points — sufficient reasoning, fit on a single GPU, Apache 2.0.

---

## Part 4: Application Use Cases

### The Three-Layer Stack

```
User query
    ↓
[LLM] — parse intent, plan retrieval, generate response
    ↓
[KG] — structured fact retrieval, multi-hop graph traversal
    ↓
[GNN] — predict missing links, score relevance, detect anomalies
    ↓
[Agent loop] — iterate, refine, take follow-up actions
```

### Use Cases by Complexity

| Use Case | Core challenge | Primary layer |
|---|---|---|
| KYC/AML investigation | Detecting suspicious structural patterns | GNN anomaly detection |
| UBO inference | Predicting missing ownership links | GNN link prediction |
| Credit risk contagion | Tracing exposure chains | KG traversal + GNN embedding |
| Regulatory Q&A | Multi-hop rule reasoning | KG + LLM |
| Investment research | Retrieval + narration | Graph RAG + LLM |
| Peer group discovery | Clustering by structural role | GNN embedding + unsupervised |

---

### KYC / AML Investigation Assistant

**The problem:** A compliance analyst gets an alert: *"Suspicious funds movement involving Entity A."* Who controls it? Are there circular ownership loops? What entities share counterparties, jurisdictions, or directors? Today this is mostly manual — hours of spreadsheet work with incomplete evidence trails.

**Why graphs are required:** Tabular ML sees rows in isolation and can flag *"Company X has risky attributes"*, but cannot detect *"Company X owns Y which owns Z which owns Company X"*. SQL can traverse relationships but cannot *learn* which structural patterns are suspicious. The GNN does both.

**System flow:**
```
Analyst: "Is there a suspicious flow between Entity A and Entity B?"
    ↓ [Agent] parses intent, identifies entity names
    ↓ [KG] retrieves the subgraph connecting A and B
    ↓ [GNN] scores each entity for anomaly likelihood
    ↓ [LLM] narrates the suspicious path in plain English
    ↓ Analyst: "Expand to entities 2 hops away from B"
    ↓ [Agent loop continues...]
```

**What each layer does:**
- **KG** — stores known facts: who owns what, who directs what, which accounts transacted with which
- **GNN** — learns relational patterns: a company with nominee directors, shell-jurisdiction registration, and circular ownership has a structural fingerprint the GNN recognizes even if no individual fact is suspicious
- **LLM** — parses analyst questions into graph queries, explains findings in plain language, maintains conversation context

**Stack:** Ownership + transaction KG → DGFraud / CARE-GNN → Llama/Mistral via Ollama → LangChain agent

**Key repos:** [DGFraud](https://github.com/safe-graph/DGFraud), [CARE-GNN](https://github.com/YingtongDou/CARE-GNN), [AWS realtime fraud](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl), [KG_RAG](https://github.com/BaranziniLab/KG_RAG)

---

### UBO Inference (Beneficial Ownership Completion)

**The problem:** FIBO and LEI data are incomplete — ownership chains, beneficial owners, and UBOs are missing or stale. Regulators require knowing who ultimately controls an entity, but the data doesn't exist.

**Why graphs are required:** Missing links in an ownership graph aren't random — they follow structural patterns. Entities that look similar in the embedding space (same jurisdiction, same relation types, similar ownership depth) tend to have similar missing links.

**System flow:**
```
Input: Known ownership + directorship triples
    ↓ [GNN link predictor] scores candidate (entity, :beneficialOwnerOf, entity) triples
    ↓ [KG] stores high-confidence predictions with provenance
    ↓ [LLM] explains the predicted chain and confidence
    ↓ Analyst reviews and approves
```

**Stack:** Ownership KG → RotatE (GraphVite) or RGCN link prediction (PyG) → LlamaIndex for analyst review interface

**Key repos:** [GraphVite](https://github.com/DeepGraphLearning/graphvite), [torch-rgcn](https://github.com/thiviyanT/torch-rgcn), [GNN-QE](https://github.com/DeepGraphLearning/GNN-QE)

---

### Credit Risk Contagion

**The problem:** *"What is our total indirect exposure to distressed entities in Region X?"* Direct exposure is easy. Second and third-order counterparty exposure is not — it requires traversing chains of relationships that are invisible in tabular data.

**System flow:**
```
Risk manager query
    ↓ [KG] identifies distressed entity cluster by jurisdiction/sector
    ↓ [GNN embedding] surfaces entities structurally close to known distressed ones
    ↓ [KG traversal] traces exposure chains: holdings → counterparties → counterparties-of-counterparties
    ↓ [LLM] aggregates exposure values, explains contagion paths, outputs risk summary
    ↓ Follow-up: "Which of these can we hedge?" → agent queries instrument nodes in KG
```

**Stack:** Counterparty exposure KG → node2vec/LINE (GraphVite) → LlamaIndex agent → Dify for workflow

**Key repos:** [FinDKG](https://github.com/vfcarida/FinDKG), [GNN-RAG](https://github.com/cmavro/GNN-RAG), [LightRAG](https://github.com/HKUDS/LightRAG)

---

### Regulatory Compliance Q&A

**The problem:** *"Does this transaction structure comply with MiFID II reporting requirements?"* Regulations are a graph — rules reference entities which are subject to obligations which have exemptions. No single rule answers the question; you need multi-hop traversal.

**System flow:**
```
Compliance query
    ↓ [KG] encodes regulation–entity–obligation triples (FIBO, LEI, MiFID II rules)
    ↓ [RGCN] reasons over heterogeneous relation types (:subjectTo, :exemptFrom, :requiresReporting)
    ↓ [LLM] retrieves applicable rules, generates compliance memo
    ↓ Agent asks clarifying questions if entity classification is ambiguous
```

**Stack:** Regulatory ontology KG (FIBO/LEI) → RGCN (PyG) → GraphRAG for broad summaries → LlamaIndex for Q&A

**Key repos:** [GNN-QE](https://github.com/DeepGraphLearning/GNN-QE), [GoG](https://github.com/YaooXu/GoG), [microsoft/graphrag](https://github.com/microsoft/graphrag), [LlamaIndex](https://github.com/run-llama/llama_index)

---

### Investment Research Chatbot

**The problem:** *"What companies does Person X control, directly or indirectly? What sectors do they cluster in? What's the counterparty exposure?"* Answers span multiple subgraphs and require both precise fact retrieval and broad narrative synthesis.

**System flow:**
```
Analyst question
    ↓ [Agent] classifies query: factual (precise traversal) vs. broad (community summary)
    ↓ [GNN-RAG] for precise multi-hop questions (ownership chains, UBO)
    ↓ [GraphRAG] for broad questions (ownership landscape summaries, sector clusters)
    ↓ [GNN embedding] for peer group / sector clustering
    ↓ [LLM] narrates, flags jurisdictions of interest, generates report
```

**Stack:** Financial RDF store → RotatE/GNN-RAG → GraphRAG + LightRAG → Llama/Mistral → LangChain agent

**Key repos:** [FinDKG](https://github.com/vfcarida/FinDKG), [microsoft/graphrag](https://github.com/microsoft/graphrag), [LightRAG](https://github.com/HKUDS/LightRAG), [GNN-RAG](https://github.com/cmavro/GNN-RAG)

---

### Recommended Stack by Use Case

| Use Case | KG | GNN | Graph RAG | Agent |
|---|---|---|---|---|
| KYC / AML | Transaction + ownership triples | DGFraud / CARE-GNN anomaly scoring | — | LangChain / CrewAI |
| UBO inference | Ownership KG | RGCN link prediction | — | LlamaIndex |
| Credit risk | Counterparty exposure graph | node2vec contagion paths | LightRAG | Dify |
| Regulatory Q&A | FIBO/LEI ontology | RGCN rule traversal | GraphRAG summaries | LlamaIndex |
| Investment research | Ownership + instrument facts | RotatE peer clustering | GraphRAG + GNN-RAG | LangChain |

---

## Part 5: GNN + LLM Hybrid Research

### GNN + LLM Hybrid Models

| Project | Repo | Venue | What it does |
|---|---|---|---|
| GOFA | [JiaruiFeng/GOFA](https://github.com/JiaruiFeng/GOFA) | ICLR 2025 | GNN layers interleaved into frozen LLM; joint graph-language pre-training |
| LLaGA | [VITA-Group/LLaGA](https://github.com/VITA-Group/LLaGA) | ICML 2024 | Graph nodes → structure-aware token sequences for LLM |
| HiGPT | [HKUDS/HiGPT](https://github.com/HKUDS/HiGPT) | KDD 2024 | Heterogeneous graph tokenizer + Mixture-of-Thought augmentation |
| GraphGPT | [HKUDS/GraphGPT](https://github.com/HKUDS/GraphGPT) | — | Dual-stage instruction tuning to align LLMs with graph structure |
| InstructGLM | [agiresearch/InstructGLM](https://github.com/agiresearch/InstructGLM) | EACL 2024 | Describes graph topology in natural language; purely generative approach |
| GraphLLM | [CurryTang/Graph-LLM](https://github.com/CurryTang/Graph-LLM) | — | Boosting LLM graph reasoning with graph context |

### Graph Foundation Models

| Project | Repo | Venue | What it does |
|---|---|---|---|
| OpenGraph | [HKUDS/OpenGraph](https://github.com/HKUDS/OpenGraph) | EMNLP 2024 | Zero-shot generalization to unseen graphs, distilled from LLMs |
| OFA (One For All) | [LechengKong/OneForAll](https://github.com/LechengKong/OneForAll) | — | Cross-domain/cross-task classification with one model |
| AnyGraph | [HKUDS/AnyGraph](https://github.com/HKUDS/AnyGraph) | — | Foundation model for diverse real-world graph types |
| GFT | [Zehong-Wang/GFT](https://github.com/Zehong-Wang/GFT) | NeurIPS 2024 | Tree vocabulary tokens; unifies node/edge/graph tasks |
| PromptGFM | [agiresearch/PromptGFM](https://github.com/agiresearch/PromptGFM) | — | Pure-language prompts; avoids out-of-vocabulary tokens |

### Domain-Specific Financial Projects

| Project | Notes |
|---|---|
| FinDKG | Dynamic financial KG from news via LLM-extracted entities; KGTransformer model |
| ChatGPT-informed GNN for stocks | ChatGPT infers network structure from news; GNN embeds companies; predicts price movement |

---

## Part 6: Curated Collections

| Collection | Repo | Focus |
|---|---|---|
| Awesome-Graph-LLM | [XiaoxinHe/Awesome-Graph-LLM](https://github.com/XiaoxinHe/Awesome-Graph-LLM) | GNN+LLM papers (NeurIPS, KDD, ACL) |
| Awesome-LLM-KG | [RManLuo/Awesome-LLM-KG](https://github.com/RManLuo/Awesome-LLM-KG) | Unifying LLMs and knowledge graphs |
| Awesome-GraphRAG | [DEEP-PolyU/Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) | Graph RAG surveys, benchmarks, implementations |
| Awesome-Foundation-Models-on-Graphs | [Zehong-Wang/Awesome-Foundation-Models-on-Graphs](https://github.com/Zehong-Wang/Awesome-Foundation-Models-on-Graphs) | Graph foundation model papers + datasets |
| GFMPapers | [BUPT-GAMMA/GFMPapers](https://github.com/BUPT-GAMMA/GFMPapers) | Must-read graph foundation model papers |
| GNNPapers | [thunlp/GNNPapers](https://github.com/thunlp/GNNPapers) | Classic must-read GNN papers |
| LLM4Graph | [SitaoLuan/LLM4Graph](https://github.com/SitaoLuan/LLM4Graph) | KDD 2024 tutorial; LLMs for graph tasks |
| Awesome-LLMs-in-Graph-tasks | [yhLeeee/Awesome-LLMs-in-Graph-tasks](https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks) | LLM utilization for graph tasks |
| KG-LLM-Papers | [zjukg/KG-LLM-Papers](https://github.com/zjukg/KG-LLM-Papers) | Comprehensive KG + LLM paper list |
