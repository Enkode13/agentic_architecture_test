# Autonomous Multi-Agent System & Production-Grade RAG Architecture

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-managed-purple.svg)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/LangGraph->=1.1.10-orange.svg)](https://github.com/langchain-ai/langgraph)
[![API Layer](https://img.shields.io/badge/FastAPI->=0.136.1-green.svg)](https://fastapi.tiangolo.com/)
[![Inference Engine](https://img.shields.io/badge/Groq-API-red.svg)](https://groq.com/)

A production-grade, asynchronous multi-agent orchestration framework built with **LangGraph**, **FastAPI**, **Milvus**, and native **Model Context Protocol (MCP)** integration. 

This repository demonstrates the progressive evolution of an LLM system from a raw Python control loop to a stateful, multi-agent architecture capable of complex routing, parallel fan-out/fan-in execution, and self-auditing tool evaluation.

---

## 🎯 Domain Context & Target Use Case

The target domain for this system is **advanced scientific Q&A over foundational quantum physics academic papers** (specifically covering Heisenberg, Schrödinger, Einstein, and Planck). 

The pipeline handles complex user queries that demand a combination of domain-specific semantic retrieval, contextual calculation, and precise tool execution using high-throughput Groq inference.

---

## 🏗️ Progressive Architecture Evolution

The workspace preserves a 3-generation chronological architecture for experimental comparison and architectural benchmarking:

```text
agentic_architecture_test/
├── 📁 core/            --> [Gen 1] Raw Python ReAct Loop (First Principles)
├── 📁 lang_core/       --> [Gen 2] Single-Agent LangGraph Pipeline
└── 📁 lang_core_multi/ --> [Gen 3] Multi-Agent Subgraph Orchestration (Active Production)
```

### 🔹 Generation 1: Manual ReAct Loop (`core/`)
- Built from first principles in **pure Python** without agent frameworks.
- Implements a custom RAG search pipeline with **hybrid search** (Dense Vector + BM25 keyword matching) and a **Cross-Encoder reranker** for retrieval precision.

### 🔹 Generation 2: Single-Agent LangGraph (`lang_core/`)
- Refactored the raw control loop into an asynchronous `StateGraph` in **LangGraph**.
- Centralized control within a single LLM router, maintaining tool modules as clean, procedural Python functions.

### 🔹 Generation 3: Multi-Agent Subgraph Orchestration (`lang_core_multi/`)
- A modular multi-agent system featuring a primary **Controller Agent** and two dedicated **Helper Subgraphs** (`Retrieval Helper` and `Tool Helper`).
- Exposes tools and subgraphs as native **Model Context Protocol (MCP)** endpoints.
- Wraps the entire engine in an asynchronous **FastAPI** event loop.

---

## 🔀 Multi-Agent Routing Logic (The 4 Routing Paths)

The main Controller LLM evaluates user intent using structured output parsing (`.with_structured_output()`) to trigger one of four distinct graph execution topologies:

```text
                          ┌──────────────────────────┐
                          │   Main Controller LLM    │
                          └─────────────┬────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  [1] Tool Only                [2] Retrieval Only           [3] Sequential
┌──────────────────┐         ┌────────────────────┐       ┌────────────────────┐
│   Tool Helper    │         │  Retrieval Helper  │       │  Retrieval Helper  │
└────────┬─────────┘         └─────────┬──────────┘       └─────────┬──────────┘
         │                             │                            │ Context Pass
         │                             │                            ▼
         │                             │                  ┌────────────────────┐
         │                             │                  │    Tool Helper     │
         │                             │                  └─────────┬──────────┘
         │                             │                            │
         └─────────────────────────────┼────────────────────────────┘
                                       ▼
                         ┌──────────────────────────┐
                         │    Consolidation Node    │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │ Main Controller Synthesis│
                         └──────────────────────────┘
```

1. **Direct Tool Route (`Tool Agent`):** Bypasses knowledge retrieval for pure computation, dictionary lookup, or tool manipulation.
2. **Direct Knowledge Route (`Retrieval Agent`):** Routes directly to the Milvus hybrid search vector database for historical paper retrieval.
3. **Sequential Dependency (`Retrieval ➔ Tool`):** Triggers when retrieved text requires subsequent computation or dictionary analysis. The context output from the Retrieval Agent feeds directly into the Tool Agent subgraph prior to consolidation.
4. **Simultaneous Fan-Out / Fan-In (`Parallel Execution`):** Triggered when a query requires both deep domain retrieval and real-time execution. The graph fans out into both the `Retrieval Agent` and `Tool Agent` subgraphs concurrently. The `consolidation_node` acts as a barrier synchronization point, merging incoming state streams before final response synthesis.

---

## 🛡️ Double-LLM Verification Loop

To prevent hallucination and incomplete tool outputs within the `Tool Helper` subgraph, execution is governed by a **dual-LLM auditor loop**:

```text
 ┌──────────────────────┐      Tool Run      ┌──────────────────────┐
 │      LLM 1:          ├───────────────────►│      Tool Node       │
 │ Executor & Answerer  │                    └──────────┬───────────┘
 └──────────────────────┘                               │ Output
            ▲                                           ▼
            │ Retry Loop                     ┌──────────────────────┐
            └────────────────────────────────┤        LLM 2:        │
                   (If Details Missing)      │ Auditor / Reviewer   │
                                             └──────────────────────┘
```

* **LLM 1 (Executor):** Determines tool parameter calls and synthesizes the preliminary output.
* **LLM 2 (Auditor):** Inspects the raw tool return against the produced answer. If information is missing or imprecise, it rejects the state update and triggers a re-execution loop.

---

## 🔌 Model Context Protocol (MCP) & Gateway Architecture

The project implements native **MCP** clients and servers (`mcp` and `FastMCP`):
- **MCP Server (`lang_core_multi/mcp/mcp_server.py`):** Exposes mathematical, dictionary, and retrieval functions as standardized, remote-callable tools over protocol endpoints.
- **MCP Client (`lang_core_multi/mcp/mcp_client.py`):** Enables agent nodes to dynamically discover and invoke server-side tools asynchronously.
- **FastAPI Gateway (`main.py`):** Wraps the entire compiled graph inside a RESTful microservice API with streaming and JSON response formats.

---

## 🛠️ Tech Stack & Key Libraries

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Package Manager** | `uv` | High-performance Python environment & lockfile management |
| **Orchestration** | `LangGraph` (`>=1.1.10`) | Stateful multi-agent subgraphs, state schema management |
| **Inference Engine** | `Groq SDK` / `ChatGroq` | Structured output extraction & low-latency LLM execution |
| **Vector Database** | `PyMilvus` (`>=2.6.12`) | Dense semantic storage and vector query search |
| **Semantic Chunking** | `semchunk` (`>=4.0.0`) | Token/character-level semantic text splitting |
| **Embeddings** | `sentence-transformers` | Local dense vector embedding generation |
| **PDF Processing** | `PyMuPDF` (`fitz`) | Academic paper ingestion & metadata parsing |
| **Protocol / Tools** | `mcp` & `FastMCP` | Decoupled client-server tool invocation protocol |
| **REST API** | `FastAPI` & `Uvicorn` | Asynchronous web server hosting the agent framework |

---

## 📁 Repository Structure

```text
agentic_architecture_test/
├── main.py                    # FastAPI application entry point; exposes HTTP endpoints
├── pyproject.toml             # Project metadata and dependency declarations (uv-managed)
├── uv.lock                    # Locked dependency versions for reproducible builds
├── .python-version            # Pinned Python interpreter version
│
├── core/                      # [Gen 1] Manual ReAct loop via raw Python (Legacy)
│   ├── agent.py               # Hand-rolled ReAct loop logic
│   ├── retrieval.py           # Hybrid vector search logic
│   ├── vectordb.py            # Milvus collection setup & query handling
│   ├── ingestion.py           # PDF ingestion pipeline
│   └── raw_docs/              # Source PDFs (Heisenberg, Schrödinger, Einstein, Planck)
│
├── lang_core/                 # [Gen 2] Single-agent LangGraph setup (Legacy)
│   ├── agent.py               # Single-agent graph compilation
│   ├── all_state.py           # State schema definitions
│   └── rag_core/              # Vector database & chunking utilities
│
└── lang_core_multi/           # [Gen 3] Multi-agent orchestration (Current System)
    ├── agent.py               # Main graph compilation, conditional edges & subgraph wiring
    ├── agent_nodes.py          # Controller, tool agent, retrieval, & consolidation nodes
    ├── agent_llm.py           # Groq SDK initialization & structured prompt mechanics
    ├── all_state.py           # Schemas: Main State, Tool State, Retrieval State
    ├── rag_core/              # Hybrid Milvus retrieval pipeline utilities
    │   ├── vectordb.py        # Milvus collection & query execution
    │   ├── chunking.py        # Semantic chunking via semchunk
    │   └── ingestion.py       # PyMuPDF ingestion pipeline
    ├── tools/                 # Execution tools (calculator, dictionary, retrieval)
    └── mcp/                   # Native MCP Server & Client implementation
        ├── mcp_server.py      # FastMCP tool server
        └── mcp_client.py      # Agent MCP client bridge
```

---

## 🚀 Getting Started

### Prerequisites

* **Python:** `3.11+`
* **Package Manager:** `uv` ([Installation Guide](https://github.com/astral-sh/uv))
* **Milvus Vector DB:** A running instance of Milvus local/standalone or Milvus Lite.
* **Groq API Key:** Required for high-throughput LLM inference.

### 1. Installation

Clone the repository and synchronize the environment using `uv`:

```bash
git clone https://github.com/your-username/agentic_architecture_test.git
cd agentic_architecture_test

# Install exact locked dependencies
uv sync
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
MILVUS_URI=http://localhost:19530
LOG_LEVEL=INFO
```

### 3. Populating the Vector Database

To ingest the quantum physics PDF papers into your Milvus collection, execute the RAG population script:

```bash
uv run python -m lang_core_multi.rag_core.ingestion
```

### 4. Running the MCP Server & API Gateway

Start the native MCP Server:

```bash
uv run python -m lang_core_multi.mcp.mcp_server
```

In a separate terminal, launch the FastAPI gateway server:

```bash
uv run uvicorn main:app --reload --port 8000
```

### 5. API Usage Example

Send a query to the multi-agent system via `curl`:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What equation did Schrödinger propose in his 1926 paper, and what is the calculated energy state when n=2?"
     }'
```

---