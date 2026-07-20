import os
import json
from groq import Groq
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
from lang_core_multi.logger import get_agent_logger
from lang_core_multi.mcp.mcp_client import mcp_registry
load_dotenv()

logger = get_agent_logger(__name__)

# API key is read from environment — never hardcoded
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── Structured output schemas ────────────────────────────────────────────────

class ToolSelection(BaseModel):
    """Single tool call decision produced by worker_llm inside the Tool Agent."""
    tool_name: str
    tool_argument: dict  # Native tool calling provides arguments as structured key-value maps


class ToolLoopDecision(BaseModel):
    """
    Decision produced after each tool execution inside the Tool Agent loop.

    more_calls_needed — True if the original task is not yet fully resolved
                        and at least one more tool call is required.
    next_tool_input   — Plain-language description of the next tool task.
                        Must be an empty string when more_calls_needed is False.
    """
    more_calls_needed: bool
    next_tool_input: str


class RetrievalDecision(BaseModel):
    """
    Decision produced by the Retrieval Worker Agent after reviewing the chunks.

    tool_needed — True if the retrieved context cannot fully answer the query
                  and a follow-up tool call is required.
    tool_input  — The exact task description to pass to the Tool Agent.
                  The Tool Agent will internally decide which tool to use.
                  Must be an empty string when tool_needed is False.
    """
    tool_needed: bool
    tool_input: str


# ── LLM wrapper ─────────────────────────────────────────────────────────────
class LLM:
    def __init__(self):
        self.client = client_groq

    # ── Controller LLM ──────────────────────────────────────────────────────
    def agent_llm(self, current_state: dict) -> dict:
        """
        Main controller brain. Reads the full orchestrator state and decides
        the next routing action, returning a JSON dict of updated State fields.
        """
        logger.info("agent_llm | ENTER | query=%r | steps=%s | scratchpad_len=%d",
                    current_state.get("query"), current_state.get("steps", "?"),
                    len(current_state.get("scratchpad", [])))

        user_prompt = f"""
Current state of the agent:

query:            {current_state["query"]}
answer:           {current_state["answer"]}
tool:             {current_state["tool"]}
tool_input:       {current_state["tool_input"]}
retrieval:        {current_state["retrieval"]}
retrieval_input:  {current_state["retrieval_input"]}
scratchpad:       {current_state["scratchpad"]}
end:              {current_state["end"]}

Decide the next action in JSON format.
"""
        system_prompt = """
You are the main controller inside a multi-agent state machine.

Your job is to decide the next action based ONLY on the current state.

Routing rules:
1. Output ONLY valid JSON. No explanations or extra text.
2. When generating text or math equations inside the "answer" field, you must strictly escape characters for JSON compliance. Ensure all double quotes are escaped and LaTeX backslashes are properly handled (e.g., use double backslashes like \\\\( or \\\\hbar).
3. Set "tool" to true if the question requires a calculation or a word definition.
4. Set "retrieval" to true if the question requires external knowledge from the vector database (quantum physics papers).
5. You may set BOTH "tool" and "retrieval" to true when both are needed simultaneously (parallel fan-out).
6. "tool_input"      must contain the exact query for the Tool Agent.
7. "retrieval_input" must contain the exact query for the Retrieval Agent.
8. "scratchpad" contains prior results — use them; do NOT repeat work already done.
9. Only set "end": true when the task is completely solved and "answer" is fully populated.

Output format (all keys required):
{
  "answer":           string,
  "tool":             boolean,
  "tool_input":       string,
  "retrieval":        boolean,
  "retrieval_input":  string,
  "end":              boolean
}
"""
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_completion_tokens=1500,
            top_p=0.7,
            stream=False,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        logger.info("agent_llm | EXIT  | tool=%s | retrieval=%s | end=%s",
                    result.get("tool"), result.get("retrieval"), result.get("end"))
        logger.debug("agent_llm | full output: %s", result)
        return result

    # ── Worker LLM — tool selector (used inside the Tool Agent loop) ─────────
#     def worker_llm(self, tool_input: str) -> ToolSelection:
#         """
#         Given a plain-language tool task, decides which specific tool to invoke
#         (calculator or dictionary) and the exact argument to pass to it.

#         This is the Tool Agent's own internal intelligence — it is always the
#         one that selects the tool. No other agent makes this decision.

#         Returns a validated ToolSelection via Groq structured output.
#         """
#         logger.info("worker_llm | ENTER | tool_input=%r", tool_input)

#         system_prompt = """
# You are a tool-selection specialist inside the Tool Agent.

# Given a task description, decide:
#   - "calculator"  if the task requires any mathematical evaluation or arithmetic.
#   - "dictionary"  if the task requires looking up the definition of a word or term.

# Return ONLY the tool name and the exact argument to pass to that tool.
# """
#         response = self.client.chat.completions.create(
#             model="openai/gpt-oss-120b",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user",   "content": f"Task: {tool_input}"},
#             ],
#             temperature=0.0,
#             stream=False,
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": {
#                     "name": "ToolSelection",
#                     "strict": True,
#                     "schema": {
#                         "type": "object",
#                         "properties": {
#                             "tool_name": {
#                                 "type": "string",
#                                 "enum": ["calculator", "dictionary"],
#                             },
#                             "tool_argument": {"type": "string"},
#                         },
#                         "required": ["tool_name", "tool_argument"],
#                         "additionalProperties": False,
#                     },
#                 },
#             },
#         )
#         raw = json.loads(response.choices[0].message.content)
#         result = ToolSelection(**raw)
#         logger.info("worker_llm | EXIT  | selected tool=%r | argument=%r",
#                     result.tool_name, result.tool_argument)
#         return result
    def worker_llm(self, tool_input: str) -> ToolSelection:
        """
        Given a task, natively selects and executes a tool from the active MCP server.
        """
        logger.info("worker_llm | ENTER | tool_input=%r", tool_input)

        # Ensure the middleware has fetched capabilities
        if not mcp_registry.tools:
            raise RuntimeError("MCP infrastructure is not initialized or no tools are available.")

        system_prompt = "You are an tool-execution specialist inside the tool agent. Use the provided tools to fulfill the user request."

        # Execute native function calling call
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Task: {tool_input}"},
                ],
                temperature=0.0,
                stream=False,
                tools=mcp_registry.tools,  
                tool_choice="auto"   
            )
        except Exception as error_trace:
            logger.error("The exact line failed because: ", exc_info=True)
            raise error_trace

        message = response.choices[0].message
        
        # Extract the native tool call properties from the API frame
        if not message.tool_calls:
            logger.warning("worker_llm | Model chose to respond directly with text. No tool called.")
            return ToolSelection(
                tool_name="direct_text_fallback",
                # Package the text explanation inside the argument payload
                tool_argument={"text_content": message.content or "No response generated."}
            )
            
        native_call = message.tool_calls[0].function
        
        # Pack the selection cleanly into the helper schema
        result = ToolSelection(
            tool_name=native_call.name,
            tool_argument=json.loads(native_call.arguments)
        )

        logger.info("worker_llm | EXIT  | selected tool=%r | arguments=%r",
                    result.tool_name, result.tool_argument)
        return result

    # ── Tool loop LLM — decides whether more tool calls are needed ───────────
    def tool_loop_llm(self, tool_input: str, tool_scratchpad: list[dict]) -> ToolLoopDecision:
        """
        Called after each tool execution inside the Tool Agent loop.

        Reviews the original task and all prior tool results in tool_scratchpad,
        then decides whether the task is fully resolved or if additional tool
        calls are still required.

        Returns a validated ToolLoopDecision via Groq structured output.
        """
        logger.info("tool_loop_llm | ENTER | tool_input=%r | completed_calls=%d",
                    tool_input, len(tool_scratchpad))

        system_prompt = """
You are a loop-controller inside the Tool Agent.

After each tool call you are given:
  - The original task.
  - All tool results collected so far.

Decide:
  - more_calls_needed: true  → additional tool calls are still required.
  - more_calls_needed: false → the task is fully resolved by the current results.
  - next_tool_input: a plain-language description of the NEXT tool task
                     (only when more_calls_needed is true, otherwise empty string).

Rules:
  - Do NOT repeat a tool call whose result already appears in the results.
  - Do NOT set more_calls_needed to true unless a concrete, different next step is required.
"""
        user_prompt = f"""
Original task: {tool_input}

Tool results so far:
{json.dumps(tool_scratchpad, indent=2)}

Should more tool calls be made?
"""
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ToolLoopDecision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "more_calls_needed": {"type": "boolean"},
                            "next_tool_input":   {"type": "string"},
                        },
                        "required": ["more_calls_needed", "next_tool_input"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        raw = json.loads(response.choices[0].message.content)
        result = ToolLoopDecision(**raw)
        logger.info("tool_loop_llm | EXIT  | more_calls_needed=%s | next_input=%r",
                    result.more_calls_needed, result.next_tool_input)
        return result

    # ── Retrieval decision LLM — retrieval worker agent's internal brain ──────
    def retrieval_decision_llm(
        self, query: str, retrieved_chunks: list[str]
    ) -> RetrievalDecision:
        """
        Called inside the Retrieval Worker Agent after the vector search completes.

        Reviews the original query and all retrieved chunks, then decides:
          - Whether the chunks alone are sufficient to answer the query, OR
          - Whether a follow-up tool call is needed, and if so, what task
            description to pass to the Tool Agent.

        The Tool Agent will independently decide which specific tool to invoke.
        This LLM's only concern is: is external tool execution needed at all,
        and if yes, what is the task?

        Returns a validated RetrievalDecision via Groq structured output.
        """
        logger.info("retrieval_decision_llm | ENTER | query=%r | chunks_retrieved=%d",
                    query, len(retrieved_chunks))

        system_prompt = """
You are the decision-maker inside a Retrieval Worker Agent.
You have retrieved context from a vector database of quantum physics papers.

Your job is to decide whether the retrieved context is sufficient to answer the query,
or whether a follow-up tool execution is needed.

A tool call is needed ONLY when:
  - The query requires a numerical calculation that the text alone cannot complete.
  - The query requires a definition or lookup that is completely missing from the chunks AND you cannot answer using your baseline knowledge.

Critical Rules:
  - If you can fully answer the user's informational or definition request using the provided context or your internal knowledge (e.g., explaining the Schrödinger Equation), set tool_needed to false.
  - Set tool_needed to true ONLY if there is a concrete, actionable task that must be processed by an external tool (like a calculator or dictionary lookup).
  - When tool_needed is true, tool_input must be a precise, self-contained task description for the Tool Agent (e.g. "calculate 6.626e-34 multiplied by 3e8"). Do NOT include the full answer text in tool_input.
  - When tool_needed is false, tool_input must be an empty string.
"""
        chunks_text = (
            "\n\n---\n\n".join(retrieved_chunks)
            if retrieved_chunks
            else "(no chunks retrieved)"
        )
        user_prompt = f"""
Query: {query}

Retrieved context:
{chunks_text}

Does this query require a follow-up tool call?
"""
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "RetrievalDecision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "tool_needed": {"type": "boolean"},
                            "tool_input":  {"type": "string"},
                        },
                        "required": ["tool_needed", "tool_input"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        raw = json.loads(response.choices[0].message.content)
        result = RetrievalDecision(**raw)
        logger.info("retrieval_decision_llm | EXIT  | tool_needed=%s | tool_input=%r",
                    result.tool_needed, result.tool_input)
        return result
