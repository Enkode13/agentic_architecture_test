# Repository & Code Structure

## Workspace Organization for Generation 3
When working on Generation 3, only look at or modify files inside the `lang_core_multi/` directory.

```text
lang_core_multi/
├── agent.py               # Main graph compilation and conditional routing edges
├── agent_nodes.py         # Concrete node function definitions (controller, consolidation, etc.)
├── agent_llm.py           # Groq SDK / ChatGroq initialization and prompt mechanics
├── all_state.py           # Holds State (main), tool_state, and retrieval_state schemas
├── rag_core/              # Hybrid Milvus retrieval pipeline utilities, no edits needed
└── tools/                 # Tool primitives
    ├── calculator.py      # math_tool
    ├── dictionary.py      # dictionary_tool
    └── retrieval.py       # retrieval_tool / Retrieval wrapper