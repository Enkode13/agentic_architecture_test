from lang_core.tools.retrieval import Retrieval, make_retrieval_tool
from lang_core.tools.calculator import math_tool
from lang_core.tools.dictionary import dictionary_tool

retrieve = Retrieval()
retrieval_tool = make_retrieval_tool(retrieve)

def build_tools():
    tools = [
        math_tool, 
        dictionary_tool, 
        retrieval_tool
        ]
    tools_by_name = {tool.name: tool for tool in tools}
    return tools, tools_by_name