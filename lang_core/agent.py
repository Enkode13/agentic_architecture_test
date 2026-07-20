from langgraph.graph import StateGraph, START, END
from lang_core.agent_llm import LLM
from lang_core.tools.registry import build_tools
from lang_core.all_state import State

max_retry=4
llm = LLM()

def controller_llm_node(state: State):
    llm_state = {
        "query":state.query,
        "input":state.input,
        "answer":state.answer,
        "tool":state.tool,
        "scratchpad":state.scratchpad,
        "end":state.end,
    }
    llm_output = llm.agent_llm(llm_state)
    if "scratchpad" in llm_output:
        del llm_output["scratchpad"]
    is_over_limit = state.steps>=(max_retry-1)
    return {
        **llm_output, 
        "steps": 1, 
        "end": llm_output.get("end", False) or is_over_limit
    }

def condition(state: State):
    print(state)
    if state.steps>=max_retry:
        print("Max steps reached!")
        return "Max Steps reached"
    elif state.end:
        return "Answer Generated"
    else:
        return "Tool Needed"

def tool_node(state: State):
    tool_name = state.tool
    tool_input = state.input
    tool = tools_by_name.get(tool_name)
    if tool is None:
        # state["answer"] = f"Unknown tool: {tool_name}"
        return {
            "scratchpad": [{"Tool_name:":tool_name, "Tool Result:": f"Unknown tool: {tool_name}"}]
            }
    
    result = tool.invoke(tool_input)
    return {
        "scratchpad": [{"Tool_name:":tool_name, "Tool Result:":result}],
    }

tools, tools_by_name = build_tools()
agent_builder = StateGraph(State)

agent_builder.add_node("controller_llm_node", controller_llm_node)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "controller_llm_node")
agent_builder.add_conditional_edges(
    "controller_llm_node",
    condition,
    {
        "Answer Generated": END,
        "Max Steps reached": END,
        "Tool Needed": "tool_node",
    }
)
agent_builder.add_edge("tool_node", "controller_llm_node")

agent_lang = agent_builder.compile()