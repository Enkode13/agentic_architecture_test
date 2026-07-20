# Technology Stack & Dependencies

## Core Orchestration & Frameworks
- **LangGraph (>=1.1.10):** Native `StateGraph` usage using explicit schema orchestration. 
  - Since this is an API layer, all node functions and conditional edges must be written as **asynchronous functions** (`async def`).
- **FastAPI (>=0.136.1) & Uvicorn (>=0.46.0):** The LangGraph engine runs inside an asynchronous FastAPI event loop. Ensure all exceptions are caught nicely to return structured JSON HTTP responses.

## Inference & Agent Intelligence
- **Groq (>=1.1.2) & LangChain Community (>=0.4.1):** Use `groq` directly for processing agent prompts.
  - Rely heavily on structured outputs for supervisor routing decisions.

## Knowledge Processing & RAG Pipeline
- **Parsing:** `pymupdf` (fitz) for reading source PDF materials.
- **Chunking:** `semchunk` (>=4.0.0) for semantic character/token splitting.
- **Embeddings:** `sentence-transformers` (>=5.3.0) for semantic vector generation.
- **Vector Database:** `pymilvus` (>=2.6.12) using the `[model]` extension for dense semantic storage and vector query search.
- **Text Utilities:** `nltk` (>=3.9.4) for processing text, tokenization, or cleaning strings before embedding.

## Implementation Guardrails
- **Environment Variables:** All Groq calls must implicitly use `os.environ.get("GROQ_API_KEY")`.
- **State Updates:** Nodes must never mutate the state object in-place; they must explicitly return a dictionary containing the updated state keys.
- **Structured Output:** For routing or intent extraction nodes, heavily utilize Groq's structured output capabilities (`.with_structured_output()`) to guarantee reliable routing types.