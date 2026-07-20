"""
agent_nodes.py — All node and condition functions for the Gen 3 multi-agent graph.

Design notes
------------
* Every I/O function is `async def` — required for the FastAPI async event loop.
* Routing/condition functions are plain synchronous `def` — LangGraph requires this.
* Nodes must return a dict of updated keys; never mutate state in-place.
* ToolState nodes run inside the Tool Agent sub-graph.
* RetrievalState nodes run inside the Retrieval Worker Agent sub-graph.
* State nodes run inside the main orchestrator graph.

Tool Agent loop topology:
    START → tool_node → tool_loop_condition ─── more calls? ──→ tool_node (loop)
                                             └── done?       ──→ END

Retrieval Worker Agent topology:
    START → retrieval_node → retrieval_to_tool_condition ─── tool needed? ──→ run_tool_agent
                                                          └── no tool?    ──→ END
"""

import asyncio
from lang_core_multi.agent_llm import LLM
from lang_core_multi.all_state import State, ToolState, RetrievalState
from lang_core_multi.tools.calculator import math_tool
from lang_core_multi.tools.dictionary import dictionary_tool
from lang_core_multi.tools.retrieval import Retrieval
from lang_core_multi.logger import get_agent_logger

# ── Module-level singletons ──────────────────────────────────────────────────
llm = LLM()
retrieval = Retrieval()
logger = get_agent_logger(__name__)

MAX_RETRY = 6          # main orchestrator step cap
TOOL_LOOP_MAX = 4      # safety cap for tool sub-graph loop iterations


# ════════════════════════════════════════════════════════════════════════════
# TOOL AGENT NODES  (operate on ToolState)
# ════════════════════════════════════════════════════════════════════════════

async def tool_node(state: ToolState) -> dict:
    """
    Single iteration of the Tool Agent loop using native tool calling.
    """
    logger.info("tool_node | ENTER | tool_steps=%d | scratchpad_len=%d",
                state.tool_steps, len(state.tool_scratchpad))

    if state.tool_scratchpad:
        last = state.tool_scratchpad[-1]
        current_input = last.get("next_tool_input") or state.tool_input
    else:
        current_input = state.tool_input

    logger.debug("tool_node | resolved current_input=%r", current_input)

    # 1. worker_llm natively selects the tool and returns the clean ToolSelection object
    selection = await asyncio.to_thread(llm.worker_llm, current_input)

    # 2. Grab the live execution function directly from the registry via the native name
    from lang_core_multi.mcp.mcp_client import mcp_registry
    # Inside tool_node right after the selection lookup:
    tool = mcp_registry.functions.get(selection.tool_name)

    if selection.tool_name == "direct_text_fallback":
        # Capture the raw text response the LLM threw out and pass it down the graph
        result_str = selection.tool_argument.get("text_content")
        logger.info("tool_node | handled direct text fallback path gracefully")
        
    elif tool is None:
        result_str = f"Unknown tool requested: {selection.tool_name}"
        logger.warning("tool_node | unknown tool requested: %s", selection.tool_name)
        
    else:
        logger.info("tool_node | invoking tool=%r with argument=%r",
                    selection.tool_name, selection.tool_argument)
        try:
            raw_result = await tool(**selection.tool_argument)
            result_str = str(raw_result)
        except Exception as e:
            logger.error(f"Execution failed inside tool node block for {selection.tool_name}", exc_info=True)
            result_str = f"Execution Error: {str(e)}"

    logger.info("tool_node | EXIT  | tool=%r | result=%r", selection.tool_name, result_str)

    return {
        "tool_name":   selection.tool_name,
        "tool_result": result_str,
        "tool_scratchpad": [{
            "tool_name":   selection.tool_name,
            "tool_result": result_str,
        }],
        "tool_steps": 1,
    }


def tool_loop_condition(state: ToolState) -> str:
    """
    Routing condition for the Tool Agent loop.

    After tool_node runs, calls tool_loop_llm to decide if the original task
    is fully resolved or if another tool call is needed.

    Returns:
        "loop"  → back to tool_node for another iteration
        "done"  → forward to END

    Safety: forces "done" if tool_steps >= TOOL_LOOP_MAX to prevent runaway loops.
    """
    logger.info("tool_loop_condition | ENTER | tool_steps=%d / max=%d",
                state.tool_steps, TOOL_LOOP_MAX)

    # Hard cap — never loop more than TOOL_LOOP_MAX times
    if state.tool_steps >= TOOL_LOOP_MAX:
        logger.warning("tool_loop_condition | step cap reached — forcing done")
        return "done"

    # Ask the loop LLM whether more calls are needed
    decision = llm.tool_loop_llm(state.tool_input, state.tool_scratchpad)

    if decision.more_calls_needed and decision.next_tool_input:
        # Write the next task into the scratchpad so tool_node can read it
        # on the next iteration.  We do this by mutating the last entry of
        # the in-memory state copy — this is safe here because LangGraph
        # passes a fresh copy to each node call and the condition function
        # runs synchronously before the next node is dispatched.
        # The actual state update happens via the return dict from tool_node;
        # here we just signal the routing decision.
        # Store next_tool_input as a transient field on ToolState via the
        # scratchpad: the next tool_node call reads tool_scratchpad[-1]["next_tool_input"].
        state.tool_scratchpad.append({"next_tool_input": decision.next_tool_input})
        logger.info("tool_loop_condition | routing → loop | next_input=%r",
                    decision.next_tool_input)
        return "loop"

    logger.info("tool_loop_condition | routing → done")
    return "done"


# ════════════════════════════════════════════════════════════════════════════
# RETRIEVAL WORKER AGENT NODES  (operate on RetrievalState)
# ════════════════════════════════════════════════════════════════════════════

async def retrieval_node(state: RetrievalState) -> dict:
    """
    Retrieval Worker Agent — phase 1: vector search.

    Executes a hybrid (dense + BM25) search against the Milvus vector store
    and stores the retrieved chunks in the state.
    """
    logger.info("retrieval_node | ENTER | retrieval_input=%r", state.retrieval_input)

    chunks = await asyncio.to_thread(
        retrieval.hybrid_search, state.retrieval_input, 5
    )

    logger.info("retrieval_node | EXIT  | chunks_retrieved=%d", len(chunks))
    logger.debug("retrieval_node | chunk preview: %s",
                 chunks[0][:120] if chunks else "(empty)")

    return {"retrieved_chunks": chunks}


async def retrieval_decision_node(state: RetrievalState) -> dict:
    """
    Retrieval Worker Agent — phase 2: internal LLM decision.

    After the chunks are in state, asks retrieval_decision_llm whether a
    follow-up tool call is needed and, if so, what the task should be.

    The Tool Agent will receive this task and internally decide which tool
    (calculator / dictionary) to invoke — that is not this node's concern.
    """
    logger.info("retrieval_decision_node | ENTER | chunks_available=%d",
                len(state.retrieved_chunks))

    decision = await asyncio.to_thread(
        llm.retrieval_decision_llm,
        state.retrieval_input,
        state.retrieved_chunks,
    )

    logger.info("retrieval_decision_node | EXIT  | tool_needed=%s | tool_input=%r",
                decision.tool_needed, decision.tool_input)

    return {
        "tool_needed": decision.tool_needed,
        "tool_input":  decision.tool_input,
    }


def retrieval_to_tool_condition(state: RetrievalState) -> str:
    """
    Routing condition inside the Retrieval Worker Agent sub-graph.

    Reads the tool_needed flag set by retrieval_decision_node and routes:
        "needs_tool" → the embedded tool sub-graph node (run_tool_agent_from_retrieval)
        "no_tool"    → END  (chunks are sufficient; agent is done)
    """
    logger.info("retrieval_to_tool_condition | tool_needed=%s", state.tool_needed)

    if state.tool_needed and state.tool_input:
        logger.info("retrieval_to_tool_condition | routing → needs_tool")
        return "needs_tool"

    logger.info("retrieval_to_tool_condition | routing → no_tool")
    return "no_tool"


# ════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR NODES  (operate on State)
# ════════════════════════════════════════════════════════════════════════════

async def controller_llm_node(state: State) -> dict:
    """
    Main controller brain.

    Calls agent_llm to decide the next routing action and returns updated
    State fields.  The step counter is incremented by 1 per call and
    accumulated by operator.add.
    """
    logger.info("controller_llm_node | ENTER | step=%d | query=%r",
                state.steps, state.query)

    llm_state = {
        "query":           state.query,
        "answer":          state.answer,
        "tool":            state.tool,
        "tool_input":      state.tool_input,
        "retrieval":       state.retrieval,
        "retrieval_input": state.retrieval_input,
        "scratchpad":      state.scratchpad,
        "end":             state.end,
    }

    llm_output = await asyncio.to_thread(llm.agent_llm, llm_state)
    llm_output.pop("scratchpad", None)

    is_over_limit = state.steps >= (MAX_RETRY - 1)

    logger.info("controller_llm_node | EXIT  | tool=%s | retrieval=%s | end=%s | over_limit=%s",
                llm_output.get("tool"), llm_output.get("retrieval"),
                llm_output.get("end"), is_over_limit)

    return {
        **llm_output,
        "steps": 1,
        "end":   llm_output.get("end", False) or is_over_limit,
    }


async def consolidation_node(state: State) -> dict:
    """
    Fan-in barrier — runs after one or both sub-agent adapter nodes finish.

    By this point the scratchpad already holds the results written by the
    adapter functions (run_tool_agent / run_retrieval_agent in agent.py).
    This node's job is to reset all routing flags so the controller starts
    its next reasoning cycle with a clean slate.
    """
    logger.info("consolidation_node | ENTER | scratchpad_len=%d", len(state.scratchpad))

    logger.info("consolidation_node | EXIT  | flags reset — routing back to controller")

    return {
        "tool":                  False,
        "retrieval":             False,
        "retrieval_tool_needed": False,
        "retrieval_tool_input":  "",
    }


def condition(state: State) -> str | list:
    """
    Main routing condition — called after controller_llm_node.

    Returns one of:
        "Answer Generated"          → END
        "Max Steps Reached"         → END
        "Tool Only"                 → run_tool_agent
        "Retrieval Only"            → run_retrieval_agent
        list of node names          → parallel fan-out to both agents
    """
    logger.info("condition | ENTER | steps=%d | tool=%s | retrieval=%s | end=%s",
                state.steps, state.tool, state.retrieval, state.end)

    if state.steps >= MAX_RETRY:
        logger.warning("condition | max steps reached (%d) → END", state.steps)
        return "Max Steps Reached"

    if state.end:
        logger.info("condition | end=True → Answer Generated")
        return "Answer Generated"

    wants_tool      = state.tool
    wants_retrieval = state.retrieval

    if wants_tool and wants_retrieval:
        logger.info("condition | both flags set → Parallel fan-out")
        # Parallel fan-out — LangGraph executes both concurrently
        return ["run_tool_agent", "run_retrieval_agent"]

    if wants_tool:
        logger.info("condition | tool=True → Tool Only")
        return "Tool Only"

    if wants_retrieval:
        logger.info("condition | retrieval=True → Retrieval Only")
        return "Retrieval Only"

    # Neither flag set — controller produced an answer or stalled
    logger.info("condition | no flags set → Answer Generated (fallback)")
    return "Answer Generated"
