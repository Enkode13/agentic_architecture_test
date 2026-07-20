from mcp.server.fastmcp import FastMCP
from lang_core_multi.tools.dictionary import dictionary_tool
from lang_core_multi.tools.calculator import math_tool

# Initialize the server instance using the core module class
mcp = FastMCP("Tools-Agent-Server")

@mcp.tool()
def calculator(expression: str) -> dict:
    """Useful for performing mathematical calculations. Accepts standard math string expressions."""
    # FastMCP reads the 'expression: str' type hint to build the LLM parameter schema
    return math_tool(expression=expression)

@mcp.tool()
async def dictionary(word: str) -> dict:
    """Useful for performing dictionary searches to find definitions of words."""
    # Works seamlessly with our async definition function
    return await dictionary_tool(word=word)

if __name__ == "__main__":
    mcp.run(transport="stdio")