"""
agent.py — Graph compilation for the Gen 3 multi-agent system.

Sub-graph topology
──────────────────

Tool Agent (ToolState):
    START → tool_node → tool_loop_condition ──"loop"──► tool_node
                                            └──"done"──► END

Retrieval Worker Agent (RetrievalState):
    START → retrieval_node → retrieval_decision_node → END
    (vector search)          (LLM: tool needed?)
    Results written to RetrievalState; main graph reads them via adapter.


Main orchestrator (State):
                        ┌──────────────────────┐
              ┌─────────│  controller_llm_node  │◄──────────────────────┐
              │         └──────────────────────┘                        │
              │                    │ condition()                         │
              │        ┌───────────┼──────────────────┐                 │
              │     Tool Only  Retrieval Only       Parallel            │
              │        │           │                  │                 │
              ▼        ▼           ▼              [fan-out]             │
       run_tool_agent  run_retrieval_agent                               │
              │        │                                                 │
              │        │  retrieval_to_tool_condition()                  │
              │        │       │              │                          │
              │    "needs_tool"│          "no_tool"                      │
              │        │       ▼              │                          │
              │        └──► run_tool_agent    │                          │
              │                  │            │                          │
              └──────────────────┴────────────┘                         │
                                 │                                       │
                       ┌─────────▼──────────┐                           │
                       │  consolidation_node │───────────────────────────┘
                       └────────────────────┘

Sequential path isolation
─────────────────────────
When the controller routes "Retrieval Only", run_retrieval_agent runs the
retrieval sub-graph, then writes retrieval_tool_needed + retrieval_tool_input
onto the main State. The conditional edge retrieval_to_tool_condition reads
those fields and either routes to run_tool_agent (tool needed) or directly
to consolidation_node (no tool needed).

The main State's `tool` / `tool_input` flags are NEVER modified by the
retrieval path. This eliminates any state conflict during parallel execution:
when both agents run concurrently, run_tool_agent writes to `tool`, while
run_retrieval_agent only writes to `retrieval_tool_needed` / `retrieval_tool_input`.
"""

from langgraph.graph import StateGraph, START, END

from lang_core_multi.all_state import State, ToolState, RetrievalState
from lang_core_multi.agent_nodes import (
    # Tool Agent nodes / conditions
    tool_node,
    tool_loop_condition,
    # Retrieval Worker Agent nodes / conditions
    retrieval_node,
    retrieval_decision_node,
    retrieval_to_tool_condition,
    # Main orchestrator nodes / conditions
    controller_llm_node,
    consolidation_node,
    condition,
)
from lang_core_multi.logger import get_agent_logger

logger = get_agent_logger(__name__)


# ── Tool Agent sub-graph ─────────────────────────────────────────────────────

tool_builder = StateGraph(ToolState)
tool_builder.add_node("tool_node", tool_node)
tool_builder.add_edge(START, "tool_node")
tool_builder.add_conditional_edges(
    "tool_node",
    tool_loop_condition,
    {
        "loop": "tool_node",
        "done": END,
    },
)
tool_agent = tool_builder.compile()


# ── Retrieval Worker Agent sub-graph ─────────────────────────────────────────
# Simple two-node pipeline: search → LLM decision → END.
# The routing decision (needs_tool / no_tool) is carried back to the main
# graph via the adapter function writing to State.

retrieval_builder = StateGraph(RetrievalState)
retrieval_builder.add_node("retrieval_node",          retrieval_node)
retrieval_builder.add_node("retrieval_decision_node", retrieval_decision_node)
retrieval_builder.add_edge(START,              "retrieval_node")
retrieval_builder.add_edge("retrieval_node",   "retrieval_decision_node")
retrieval_builder.add_edge("retrieval_decision_node", END)
retrieval_agent = retrieval_builder.compile()


# ── Main orchestrator adapter functions ──────────────────────────────────────

async def run_tool_agent(state: State) -> dict:
    """
    Runs the Tool Agent sub-graph (with its internal loop).

    Input:  state.tool_input  — either set by the controller (Tool Only path)
                                or by run_retrieval_agent (sequential path via
                                retrieval_tool_input).
    Output: appends a full tool result entry to the main scratchpad.
            Clears the `tool` flag.
    """
    # When arriving from the sequential path, the task lives in
    # retrieval_tool_input; for the direct tool path it is in tool_input.
    effective_input = state.retrieval_tool_input or state.tool_input

    logger.info("run_tool_agent | ENTER | effective_input=%r", effective_input)

    sub_input  = ToolState(tool_input=effective_input)
    sub_output = await tool_agent.ainvoke(sub_input)

    logger.info("run_tool_agent | EXIT  | loop_iterations=%d | final_result=%r",
                len(sub_output.get("tool_scratchpad", [])), sub_output.get("tool_result", ""))

    return {
        "scratchpad": [{
            "agent":           "tool_agent",
            "tool_scratchpad": sub_output.get("tool_scratchpad", []),
            "final_result":    sub_output.get("tool_result", ""),
        }],
        "tool":                  False,
        # Clear the retrieval-to-tool signal so it does not persist
        "retrieval_tool_needed": False,
        "retrieval_tool_input":  "",
    }


async def run_retrieval_agent(state: State) -> dict:
    """
    Runs the Retrieval Worker Agent sub-graph.

    Input:  state.retrieval_input
    Output: appends retrieved chunks to the main scratchpad.
            Writes retrieval_tool_needed + retrieval_tool_input onto State
            so retrieval_to_tool_condition can route correctly.
            Clears the `retrieval` flag.

    The main State's `tool` / `tool_input` flags are never touched here,
    preventing any conflict during parallel execution.
    """
    logger.info("run_retrieval_agent | ENTER | retrieval_input=%r", state.retrieval_input)

    sub_input  = RetrievalState(retrieval_input=state.retrieval_input)
    sub_output = await retrieval_agent.ainvoke(sub_input)

    logger.info("run_retrieval_agent | EXIT  | chunks=%d | tool_needed=%s | tool_input=%r",
                len(sub_output.get("retrieved_chunks", [])),
                sub_output.get("tool_needed", False),
                sub_output.get("tool_input", ""))

    return {
        "scratchpad": [{
            "agent":            "retrieval_agent",
            "retrieved_chunks": sub_output.get("retrieved_chunks", []),
        }],
        "retrieval":             False,
        # Signal fields read by retrieval_to_tool_condition
        # "retrieval_tool_needed": sub_output.get("tool_needed", False),
        # "retrieval_tool_input":  sub_output.get("tool_input",  ""),
    }


# ── Routing condition for post-retrieval edge ────────────────────────────────

def retrieval_to_tool_condition_main(state: State) -> str:
    """
    Conditional edge attached to run_retrieval_agent in the main graph.

    Reads retrieval_tool_needed from State (written by run_retrieval_agent)
    and routes:
        "needs_tool" → run_tool_agent
        "no_tool"    → consolidation_node
    """
    if state.retrieval_tool_needed and state.retrieval_tool_input:
        logger.info("retrieval_to_tool_condition_main | routing → run_tool_agent")
        return "needs_tool"
    logger.info("retrieval_to_tool_condition_main | routing → consolidation_node")
    return "no_tool"


# ── Main orchestrator graph ───────────────────────────────────────────────────

agent_builder = StateGraph(State)

agent_builder.add_node("controller_llm_node", controller_llm_node)
agent_builder.add_node("run_tool_agent",       run_tool_agent)
agent_builder.add_node("run_retrieval_agent",  run_retrieval_agent)
agent_builder.add_node("consolidation_node",   consolidation_node)

agent_builder.add_edge(START, "controller_llm_node")

# ── Controller routing ────────────────────────────────────────────────────────
# condition() returns a string token or a list (for parallel fan-out).
agent_builder.add_conditional_edges(
    "controller_llm_node",
    condition,
    {
        "Answer Generated":  END,
        "Max Steps Reached": END,
        "Tool Only":         "run_tool_agent",
        "Retrieval Only":    "run_retrieval_agent",
        "run_tool_agent":      "run_tool_agent",
        "run_retrieval_agent": "run_retrieval_agent",
    },
)

# ── Post-retrieval conditional edge ──────────────────────────────────────────
# After run_retrieval_agent finishes, decide whether to route to the
# Tool Agent (sequential dependency) or directly to consolidation.
agent_builder.add_conditional_edges(
    "run_retrieval_agent",
    retrieval_to_tool_condition_main,
    {
        "needs_tool": "run_tool_agent",
        "no_tool":    "consolidation_node",
    },
)

# ── Tool agent always goes to consolidation ───────────────────────────────────
agent_builder.add_edge("run_tool_agent", "consolidation_node")

# ── Consolidation loops back to controller ────────────────────────────────────
agent_builder.add_edge("consolidation_node", "controller_llm_node")

# ── Compile ───────────────────────────────────────────────────────────────────
agent_lang = agent_builder.compile()
