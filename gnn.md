# GNN + LLM Open Source Reference

---

## Part 1: GraphVite with Financial RDF Data

### Knowledge Graph Embedding

Your triples likely look like:
```
:CompanyA  :owns              :CompanyB
:CompanyA  :headquarteredIn   :Jurisdiction_X
:PersonA   :directorOf        :CompanyB
:CompanyA  :counterpartyOf    :CompanyC
:LoanA     :issuedTo          :CompanyA
:LoanA     :securedBy         :AssetX
```

#### Fraud / AML detection
**Problem:** Shell company networks, circular ownership, layered transactions are hard to spot in tabular data.
**How:** Train RotatE on ownership + transaction triples. Entities involved in suspicious structures (circular flows, unusual relation combinations) will have anomalous embeddings — flag them as outliers or use embeddings as features in a fraud classifier.

#### Counterparty / credit risk
**Problem:** Predict which counterparties are likely to default or become distressed.
**How:** Link prediction over triples like `(:Company :hasExposureTo :Company)`. The model learns that companies sharing certain relation patterns (same jurisdiction, same sector, same creditor) cluster together — surface implicit risk propagation paths.

#### Regulatory knowledge base completion
**Problem:** FIBO / LEI data is incomplete — ownership chains, beneficial owners, and UBOs (Ultimate Beneficial Owners) are missing or stale.
**How:** Train on known ownership triples, predict missing `:beneficialOwnerOf`, `:controlledBy` links. Directly actionable for KYC/AML compliance workflows.

#### Entity resolution across sources
**Problem:** Same company appears differently across Bloomberg, Refinitiv, SEC EDGAR, and your internal RDF store.
**How:** Merge RDF graphs from multiple sources, train embeddings, find entities whose embeddings are nearest neighbors across source-specific subgraphs.

---

### Node Embedding (projected graph)

Flatten relations, keep only entity-entity edges. Useful when you care about network topology, not relation types.

#### Systemic risk / contagion modeling
**Problem:** Which entities are systemically important — whose failure propagates most widely?
**How:** Project to homogeneous graph, train node2vec. High-centrality nodes in embedding space = high systemic importance. More nuanced than PageRank because it captures second/third-order structural role, not just degree.

#### Sector / peer group discovery
**Problem:** Standard SIC/NAICS codes are too coarse; you want peer groups defined by actual financial relationships.
**How:** Cluster node embeddings on a co-investment or co-counterparty graph. Emergent clusters often reveal undisclosed sector exposure or shadow peer groups invisible in metadata.

#### Portfolio correlation
**Problem:** Two assets look uncorrelated by price history but are structurally linked through common counterparties or ownership.
**How:** Build graph from shared-entity relations, embed, use embedding cosine similarity as a structural correlation signal alongside price correlation.

---

### Visualization

#### Ownership network exploration
Plot entity embeddings colored by `rdf:type` (`:Company`, `:Person`, `:Jurisdiction`, `:Instrument`). Shell company clusters and hub nodes (lawyers, nominee directors appearing in many ownership chains) stand out visually.

#### Audit / data quality
Flag entities that land in the wrong cluster — e.g. a `:NaturalPerson` that embeds near `:ShellCompany` nodes suggests it's a nominee director or a data error.

---

### Recommended model by task (GraphVite)

| Task | Model | Why |
|---|---|---|
| AML / fraud detection | RotatE | Handles antisymmetric relations (`:ownerOf` ≠ reverse) |
| KB completion / UBO inference | RotatE or ComplEx | Best accuracy on asymmetric relations |
| Systemic risk topology | node2vec | BFS/DFS balance captures both local and global structure |
| Peer group / sector clustering | LINE | Explicit 2nd-order proximity = shared-neighbor grouping |
| Visualization / audit | LargeVis on KG embeddings | After RotatE, visualize entity space |

---

## Part 2: Open Source GNN Ecosystem

### GNN Frameworks

| Framework | GitHub | Stars | Backend | Key Algorithms |
|---|---|---|---|---|
| PyTorch Geometric (PyG) | pyg-team/pytorch_geometric | ~15.9k | PyTorch | GCN, GAT, GraphSAGE, GIN, RGCN, HGT, GraphTransformer |
| Deep Graph Library (DGL) | dmlc/dgl | ~13.7k | PyTorch / TF / MXNet | GCN, GAT, GraphSAGE, GIN, RGCN, GATv2 |
| Spektral | danielegrattarola/spektral | — | Keras / TF 2 | GCN, GAT, GraphSAGE, spectral methods |
| GraphVite | DeepGraphLearning/graphvite | — | PyTorch (CPU+GPU) | DeepWalk, LINE, node2vec, TransE, RotatE, LargeVis |
| GraphGym | snap-stanford/GraphGym | — | PyTorch | Design space search over GNN architectures |

---

### GNN Model Implementations

| Model | GitHub | Notes |
|---|---|---|
| GAT (Graph Attention Network) | PetarV-/GAT | Official TF implementation |
| GAT (PyTorch) | gordicaleksa/pytorch-GAT | ~2.6k stars, tutorial-quality |
| GraphSAGE | williamleif/GraphSAGE | Official implementation |
| GIN (Graph Isomorphism Network) | yukiTakezawa/GraphIsomorphismNetwork | PyTorch |
| RGCN | tkipf/relational-gcn | Official Keras implementation |
| RGCN (PyTorch) | thiviyanT/torch-rgcn | PyTorch port |
| HGT (Heterogeneous Graph Transformer) | acbull/pyHGT | PyG version |
| HGT (DGL) | acbull/HGT-DGL | DGL version |
| PyG Temporal | benedekrozemberczki/pytorch_geometric_temporal | Spatiotemporal / dynamic graphs |
| Multi-model suite | dsgiitr/graph_nets | DeepWalk, GCN, GraphSAGE, ChebNet, GAT |

---

### GNN Applications

#### Fraud / Anomaly Detection
| Project | GitHub | Notes |
|---|---|---|
| DGFraud | safe-graph/DGFraud | Deep graph fraud detection toolbox |
| CARE-GNN | YingtongDou/CARE-GNN | Against camouflaged fraudsters (CIKM 2020) |
| AWS Realtime Fraud Detection | awslabs/realtime-fraud-detection-with-gnn-on-dgl | DGL + Amazon Neptune + SageMaker |
| Graph Fraud Detection Papers | safe-graph/graph-fraud-detection-papers | Curated paper list |

#### Knowledge Graphs
| Project | GitHub | Notes |
|---|---|---|
| GNN-QE | DeepGraphLearning/GNN-QE | Logical query execution (ICML 2022) |
| SE-GNN | renli1024/SE-GNN | Semantic evidence KG embedding (AAAI 2022) |

#### Recommendation Systems
| Project | GitHub | Notes |
|---|---|---|
| GNN-Recommender-Systems | tsinghua-fib-lab/GNN-Recommender-Systems | Algorithm index |
| DR-GNN | WANGBohaO-jpg/DR-GNN | Distributionally robust (WWW 2024) |
| GNN-RecSys | je-dbl/GNN-RecSys | DGL-based |

#### Drug Discovery / Life Sciences
| Project | GitHub | Notes |
|---|---|---|
| DGL-LifeSci | awslabs/dgl-lifesci | Molecular property prediction, generation, reaction |

#### Traffic
| Project | GitHub | Notes |
|---|---|---|
| GNN4Traffic | jwwthu/GNN4Traffic | Traffic forecasting survey + implementations |

---

## Part 3: Open Source LLM Models

### Model Families

| Family | Org | Sizes | License | Highlights |
|---|---|---|---|---|
| Llama 3 / 3.1 / 3.2 / 4 | Meta | 1B–405B (+ MoE) | Apache 2.0 | 128K context (3.1), MoE in Llama 4 |
| Mistral / Mistral Small | Mistral AI | 7B–123B | Apache 2.0 | Strong coding, instruction following |
| Qwen 2 / 3 / 3.5 | Alibaba | 0.5B–235B MoE | Apache 2.0 | Top multilingual, 201 languages (3.5) |
| Gemma 2 / 3 / 4 | Google | 1B–31B | Apache 2.0 | Memory-efficient; Gemma 3 4B = 4.2 GB RAM |
| Phi-3 / Phi-4 | Microsoft | 3.8B–14B | MIT | Strong reasoning per parameter |
| DeepSeek V3 / R1 | DeepSeek | 7B–671B MoE | MIT | R1 competitive with OpenAI o1 on math/code |
| Yi 1.5 | 01.AI | 6B–34B | Apache 2.0 | Bilingual English-Chinese |
| Falcon | TII | 7B–180B | Apache 2.0 | General purpose |
| ChatGLM | Tsinghua | 3B–130B | MIT | Chinese-optimized |
| InternLM | Shanghai AI Lab | 7B–104B | Apache 2.0 | Instruction + code |

---

## Part 4: LLM Infrastructure

### Inference & Serving

| Framework | GitHub | License | Best For |
|---|---|---|---|
| vLLM | vllm-project/vllm | Apache 2.0 | High-throughput production, multi-GPU (35–44× vs llama.cpp) |
| llama.cpp | ggerganov/llama.cpp | MIT | Portable CPU/GPU inference, edge devices |
| Ollama | ollama/ollama | MIT | Local prototyping, Docker-style UX |
| TensorRT-LLM | NVIDIA/TensorRT-LLM | Apache 2.0 | Maximum NVIDIA GPU throughput |
| SGLang | sgl-project/sglang | Apache 2.0 | Structured generation, RAG |
| text-generation-webui | oobabooga/text-generation-webui | AGPL-3.0 | Interactive experimentation, web UI |

### Fine-tuning

| Framework | GitHub | License | Key Methods |
|---|---|---|---|
| Axolotl | axolotl-ai-cloud/axolotl | Apache 2.0 | LoRA, QLoRA, DPO, GRPO, FSDP/DeepSpeed |
| LLaMA-Factory | hiyouga/LlamaFactory | Apache 2.0 | 100+ models, Unsloth integration |
| Unsloth | unslothai/unsloth | MIT | 2× faster, 70% less VRAM, 500+ models |
| TRL | huggingface/trl | Apache 2.0 | RLHF, DPO, KTO, ORPO, GRPO |
| DeepSpeed | microsoft/DeepSpeed | Apache 2.0 | Multi-node, ZeRO optimization |
| TorchTune | pytorch/torchtune | BSD | PyTorch-native, composable |

### RAG & Agent Frameworks

| Project | GitHub | Stars | Notes |
|---|---|---|---|
| LangChain | langchain-ai/langchain | ~90k | Agent orchestration, RAG chains |
| LlamaIndex | run-llama/llama_index | ~38k | Data framework for LLM apps |
| Dify | langgenius/dify | ~40k | Visual workflow builder, RAG pipelines |
| RAGFlow | infiniflow/ragflow | ~15k | Enterprise RAG engine + agents |
| CrewAI | joaomdmoura/crewai | ~51k | Multi-agent role-playing |
| AutoGPT | Significant-Gravitas/AutoGPT | ~166k | Goal-driven autonomous agents |
| OpenHands | All-Hands-AI/OpenHands | ~72k | Software development agents |

---

## Part 5: GNN + LLM Hybrid Projects

### Graph RAG

| Project | GitHub | Notes |
|---|---|---|
| GraphRAG | microsoft/graphrag | Extracts KG from text, community hierarchies, multi-hop QA |
| LightRAG | lightrag.github.io | Dual-level retrieval, graph-vector hybrid, cost-efficient |
| GeAR | arxiv 2412.18431 | Graph expansion + multi-step retrieval agent (ACL 2025, >10% on MuSiQue) |
| Awesome-GraphRAG | DEEP-PolyU/Awesome-GraphRAG | Curated collection |

### GNN + LLM Hybrid Models

| Project | GitHub | Venue | What it does |
|---|---|---|---|
| GraphGPT | graphgpt.github.io | — | Aligns LLMs with graph structure via dual-stage instruction tuning |
| InstructGLM | agiresearch/InstructGLM | EACL 2024 | Describes graph structure in natural language for generative LLM |
| LLaGA | VITA-Group/LLaGA | ICML 2024 | Graph nodes → structure-aware token sequences fed to LLM |
| HiGPT | HKUDS/HiGPT | KDD 2024 | Heterogeneous graph tokenizer + Mixture-of-Thought augmentation |
| GOFA | JiaruiFeng/GOFA | ICLR 2025 | GNN layers interleaved into frozen LLM; joint graph-language pre-training |
| GraphLLM | CurryTang/Graph-LLM | IEEE TBD | Boosting LLM graph reasoning |

### Graph Foundation Models

| Project | GitHub | Venue | What it does |
|---|---|---|---|
| OpenGraph | HKUDS/OpenGraph | EMNLP 2024 | Zero-shot graph generalization distilled from LLMs |
| GFT | Zehong-Wang/GFT | NeurIPS 2024 | Tree vocabulary tokens; unifies node/edge/graph tasks |
| AnyGraph | HKUDS/AnyGraph | — | Foundation model for diverse real-world graphs |
| OFA (One For All) | LechengKong/OneForAll | — | Cross-domain/cross-task classification with one model |
| PromptGFM | agiresearch/PromptGFM | — | Pure-language prompts; avoids out-of-vocabulary tokens |

### Knowledge Graph + LLM

| Project | GitHub | Venue | What it does |
|---|---|---|---|
| GNN-RAG | arxiv 2405.20139 | — | GNN retrieval + LLM reasoning; SOTA on KGQA, beats GPT-4 with 7B |
| KG_RAG | BaranziniLab/KG_RAG | — | Task-agnostic KG + LLM for knowledge-intensive tasks |
| GoG | YaooXu/GoG | EMNLP 2024 | LLM as both agent and KG for incomplete KGQA |
| text2graph_llm | UW-xDD/text2graph_llm | — | LLM extracts (subject, predicate, object) triplets from text |
| KG-LLM-Papers | zjukg/KG-LLM-Papers | — | Comprehensive paper list |

### Domain-Specific GNN + LLM

#### Financial
| Project | Notes |
|---|---|
| FinDKG | Dynamic financial KG from news via LLM-extracted entities; KGTransformer model |
| ChatGPT-informed GNN for stocks | ChatGPT infers network structure from news; GNN embeds companies; predicts movement |

#### Medical / Biomedical
| Project | Notes |
|---|---|
| medIKAL | KG as LLM assistant for clinical diagnosis on EMRs |
| CoMed | LoRA-tuned LLaMA + heterogeneous GNN fused for EHR prediction |

#### Molecular / Chemistry
| Project | Venue | Notes |
|---|---|---|
| LaMGen | Nature Communications | LLM-based 3D molecular generation for multi-target drug design |
| Llamole | — | LLM gatekeeper + graph diffusion model + GNN encoder |
| LLaMo | NeurIPS 2024 | Large language model-based molecular graph assistant |

#### Recommendation
| Project | Notes |
|---|---|
| RecMind | LLM adapters + GNN interaction graph + contrastive alignment + fusion gate |

---

## Part 6: Agentic AI Application Use Cases (KG + GNN + LLM)

The combination works as a three-layer stack:

```
User query
    ↓
[LLM] — parse intent, plan retrieval steps, generate response
    ↓
[KG] — structured fact retrieval, multi-hop graph traversal
    ↓
[GNN] — predict missing links, score entity relevance, detect anomalies
    ↓
[Agent loop] — iterate, refine, take follow-up actions
```

---

### Financial Chatbot / Investment Research Agent

**Stack:** Financial RDF store (ownership, instruments, filings) + RotatE/GNN-RAG + LLM (Llama/Mistral) + LangChain agent

**What it does:**
- User asks: *"What companies does Person X control, directly or indirectly?"*
- Agent traverses `:ownerOf`, `:directorOf`, `:controlledBy` chains in the KG
- GNN predicts likely-but-missing ownership links (UBO inference)
- LLM narrates the ownership chain and flags jurisdictions of interest
- Multi-turn: agent can expand to counterparty exposure, sector clustering

**Key projects:** FinDKG, GNN-RAG, GraphRAG, LangChain

---

### KYC / AML Investigation Assistant

**Stack:** Transaction + ownership KG + fraud GNN (DGFraud / CARE-GNN) + LLM + agentic loop

**What it does:**
- Compliance officer asks: *"Is there a suspicious flow between Entity A and Entity B?"*
- Agent maps query to KG entities, retrieves transaction subgraph
- GNN scores entity pairs for anomalous structural patterns (circular flows, layering)
- LLM explains the suspicious path in plain language and suggests next investigative steps
- Agent can autonomously expand the search: *"Also check entities 2 hops away"*

**Key projects:** DGFraud, CARE-GNN, AWS realtime-fraud-detection-with-gnn-on-dgl, KG_RAG

---

### Regulatory Compliance Q&A Agent

**Stack:** Regulatory ontology KG (FIBO, LEI, GDPR rules) + RGCN + LLM + RAG

**What it does:**
- User asks: *"Does this transaction structure comply with MiFID II reporting requirements?"*
- KG encodes regulation–entity–obligation triples
- RGCN reasons over heterogeneous relation types (`:subjectTo`, `:exemptFrom`, `:requiresReporting`)
- LLM retrieves and interprets applicable rules, generates a compliance memo
- Agent can ask clarifying questions if entity classification is ambiguous

**Key projects:** GNN-QE, GoG, GraphRAG, LlamaIndex

---

### Credit Risk Assessment Chatbot

**Stack:** Counterparty exposure KG + node2vec/LINE + LLM + Dify or LlamaIndex

**What it does:**
- Risk manager asks: *"What is our total indirect exposure to distressed entities in Region X?"*
- Agent identifies distressed entity cluster via GNN embeddings (anomaly outliers)
- KG traces exposure chains: direct holdings → counterparties → counterparties of counterparties
- LLM aggregates exposure values, explains contagion paths, outputs a risk summary table
- Follow-up: *"Which of these can we hedge?"* — agent queries instrument nodes in KG

**Key projects:** GraphRAG, GNN-RAG, LlamaIndex agent, FinDKG

---

### Fraud / Dispute Resolution Agent

**Stack:** Transaction graph KG + heterogeneous GNN + LLM + multi-step agent (OpenHands / CrewAI)

**What it does:**
- Investigator submits a case: *"Flag all accounts involved in this transaction cluster"*
- Agent uses GNN to score every connected entity for fraud likelihood
- KG provides context: account type, jurisdiction, prior flags, linked persons
- LLM writes a case narrative with evidence trail
- Multi-agent variant: one agent investigates the financial graph, another cross-checks identity records

**Key projects:** DGFraud, CARE-GNN, CrewAI, OpenHands

---

### Knowledge Base Completion + Chatbot (General)

**Stack:** Domain KG (any RDF store) + RotatE/ComplEx + LLM + GraphRAG

**What it does:**
- Chatbot answers questions about entities in the KG
- GraphRAG builds community summaries over the KG for broad questions (*"Summarize the ownership landscape in Jurisdiction X"*)
- GNN-RAG handles precise multi-hop factual questions (*"Who is the ultimate beneficial owner of Company Y?"*)
- LLM handles open-ended questions that go beyond the KG using its parametric knowledge
- Agent decides which retrieval path to use based on query type

**Key projects:** microsoft/graphrag, GNN-RAG, KG_RAG, GoG, LightRAG

---

### Recommended Stack by Use Case

| Use Case | KG Role | GNN Role | LLM Role | Agent Framework |
|---|---|---|---|---|
| Investment research | Ownership / instrument facts | Peer group clustering | Report generation | LangChain |
| KYC / AML | Transaction + ownership triples | Anomaly scoring, link prediction | Investigation narrative | CrewAI |
| Regulatory Q&A | Rule–entity–obligation ontology | Multi-hop rule traversal | Compliance memo | LlamaIndex |
| Credit risk | Counterparty exposure graph | Contagion path detection | Risk summary | Dify |
| Fraud investigation | Transaction subgraph | Fraud entity scoring | Case narrative | OpenHands |
| General KG chatbot | Any domain RDF store | KB completion | Answer generation | GraphRAG + LlamaIndex |

---

## Part 7: Curated Collections

| Collection | GitHub | Focus |
|---|---|---|
| Awesome-Graph-LLM | XiaoxinHe/Awesome-Graph-LLM | GNN+LLM papers (NeurIPS, KDD, ACL) |
| Awesome-LLM-KG | RManLuo/Awesome-LLM-KG | Unifying LLMs and knowledge graphs |
| Awesome-GraphRAG | DEEP-PolyU/Awesome-GraphRAG | Graph RAG surveys, benchmarks, implementations |
| Awesome-Foundation-Models-on-Graphs | Zehong-Wang/Awesome-Foundation-Models-on-Graphs | Graph foundation model papers + datasets |
| GFMPapers | BUPT-GAMMA/GFMPapers | Must-read graph foundation model papers |
| GNNPapers | thunlp/GNNPapers | Classic must-read GNN papers |
| LLM4Graph | SitaoLuan/LLM4Graph | KDD 2024 tutorial; LLMs for graph tasks survey |
| Awesome-LLMs-in-Graph-tasks | yhLeeee/Awesome-LLMs-in-Graph-tasks | LLM utilization for graph tasks |
| Awesome-GNN-in-LLMs-Papers | xkLi-Allen/Awesome-GNN-in-LLMs-Papers | GNN integration inside LLMs |
