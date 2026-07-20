from pydantic import BaseModel, Field
from typing import Annotated
import operator


# ── Main orchestrator state ──────────────────────────────────────────────────
class State(BaseModel):
    # The original user question — never mutated after entry
    query: str = ""

    # Controller output fields
    answer: str = ""
    tool: bool = False
    tool_input: str = ""
    retrieval: bool = False
    retrieval_input: str = ""

    # Accumulated results from all agents across all steps.
    # Annotated with operator.add so LangGraph merges lists rather than overwrites.
    scratchpad: Annotated[list[dict], operator.add] = Field(default_factory=list)

    # Step counter — each node increments by returning {"steps": 1}.
    # LangGraph accumulates via operator.add.
    steps: Annotated[int, operator.add] = 0

    # True when the controller has produced a final answer or the step limit hit
    end: bool = False

    # ── Retrieval-to-tool signal fields ──────────────────────────────────────
    # Written by run_retrieval_agent after its sub-graph finishes.
    # Read by retrieval_to_tool_condition to decide whether to route directly
    # to the tool sub-graph or to consolidation_node.
    # Kept separate from the main `tool` / `tool_input` flags so there is
    # zero state conflict when both agents run in parallel.
    retrieval_tool_needed: bool = False
    retrieval_tool_input: str = ""


# ── Tool Agent sub-graph state ───────────────────────────────────────────────
class ToolState(BaseModel):
    # The task description passed in — either from the main controller
    # or forwarded from the retrieval worker agent.
    tool_input: str = ""

    # Accumulated tool call results within this sub-graph's own loop.
    # Each entry is {"tool_name": ..., "tool_result": ...}.
    # Annotated with operator.add so LangGraph appends rather than overwrites
    # across loop iterations.
    tool_scratchpad: Annotated[list[dict], operator.add] = Field(default_factory=list)

    # The tool most recently selected and the result it produced.
    # Overwritten on every loop iteration.
    tool_name: str = ""
    tool_result: str = ""

    # Set to True by the tool loop condition when all required tool calls
    # are complete, which causes the sub-graph to exit to END.
    tool_done: bool = False

    # Loop safety cap — incremented by 1 per tool_node call.
    tool_steps: Annotated[int, operator.add] = 0


# ── Retrieval Agent sub-graph state ─────────────────────────────────────────
class RetrievalState(BaseModel):
    # The semantic query passed in from the main controller
    retrieval_input: str = ""

    # Populated by retrieval_node after hybrid search
    retrieved_chunks: list = Field(default_factory=list)

    # Populated by the retrieval worker LLM after reviewing the chunks.
    # True  → the retrieval result requires a follow-up tool call.
    # False → the chunks are sufficient on their own.
    tool_needed: bool = False

    # If tool_needed is True, this is the exact input the Tool Agent should receive.
    tool_input: str = ""
