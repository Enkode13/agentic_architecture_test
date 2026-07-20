# mcp_client.py
import os
import sys
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lang_core_multi.logger import get_agent_logger

logger = get_agent_logger(__name__)

class MCPRegistry:
    def __init__(self):
        self.tools = []
        self.functions = {}

mcp_registry = MCPRegistry()

native_llm_tools = mcp_registry.tools
functions_dict = mcp_registry.functions

# Global tracking blocks managed by the infrastructure layer
_exit_stack = AsyncExitStack()
_session: ClientSession | None = None

def _make_tool_proxy(tool_name: str):
    """
    Factory closure that wraps the session execution channel.
    Injects clear error interception and centralized contextual logging.
    """
    async def tool_proxy(**kwargs) -> str:
        if not _session:
            raise RuntimeError("MCP Client Session is not active.")
            
        logger.info(f"Executing tool: {tool_name} | Payloads: {kwargs}")
        
        try:
            # Send the JSON-RPC call over the active stdio loop
            result = await _session.call_tool(tool_name, arguments=kwargs)
            
            # Look for explicit runtime flags inside the protocol envelope
            if hasattr(result, 'isError') and result.isError:
                logger.error(f"Tool execution error reported by server for {tool_name}")
                return f"Execution Error: {result.content[0].text}"
                
            return result.content[0].text
            
        except Exception as e:
            # Captures low-level pipeline connection drops, timeouts, or parsing crashes
            logger.error(f"Fatal protocol exception hit while running tool: {tool_name}", exc_info=True)
            return f"Protocol Error: {str(e)}"
            
    # Preserve the signature name to assist in tracking stacks
    tool_proxy.__name__ = tool_name
    return tool_proxy

async def initialize_mcp_infrastructure():
    """
    Launches the subprocess, binds communication handles, and automatically
    builds the tracking maps and native schemas. Logs hidden server traces.
    """
    global _session
    logger.info("Spawning FastMCP server background infrastructure...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "mcp_server.py")

    # 1. Dynamically locate the absolute path to your active virtual environment directory
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")

    # Fallback to sys.executable if for some reason the .venv folder structure is different
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    # 2. Add the project root directly to the PYTHONPATH environment map
    custom_env = os.environ.copy()
    custom_env["PYTHONPATH"] = project_root

    server_params = StdioServerParameters(
        command=venv_python,        # Force execution via your project's local virtual environment binary
        args=["-u", server_path],   # Unbuffered streams ensure messages pass through instantly
        env=custom_env
    )

    # 1. Open background process hooks
    context_manager = stdio_client(server_params)
    read_stream, write_stream = await _exit_stack.enter_async_context(context_manager)

    # ── DIAGNOSTIC HOOK ──────────────────────────────────────────────────────
    if hasattr(context_manager, "_process") and context_manager._process:
        process = context_manager._process
        
        async def log_server_errors():
            try:
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    logger.error(f"[SERVER CRASH LOG]: {line.decode().strip()}")
            except Exception as e:
                logger.error(f"Diagnostics error reading stderr: {e}")
                
        # Fire the stderr listening loop completely in the background
        asyncio.create_task(log_server_errors())
    # ─────────────────────────────────────────────────────────────────────────

    # 2. Establish session context
    _session = await _exit_stack.enter_async_context(
        ClientSession(read_stream, write_stream)
    )
    
    logger.info("Sending protocol handshake to server...")
    try:
        await _session.initialize()
    except Exception as e:
        logger.error("Handshake failed! Check the [SERVER CRASH LOG] prints above.")
        raise e

    # 3. Harvest raw protocol declarations
    response = await _session.list_tools()
    
    # 4. Clean structural JSON Schema fields to prevent LLM validation errors
    cleaned_tools = []
    for tool in response.tools:
        schema = dict(tool.inputSchema)
        schema.pop("$schema", None)
        schema.pop("additionalProperties", None)
        
        cleaned_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema
            }
        })

    # 5. Mutate the stable registry references in place
    mcp_registry.functions.clear()
    mcp_registry.functions.update({tool.name: _make_tool_proxy(tool.name) for tool in response.tools})
    
    mcp_registry.tools.clear()
    mcp_registry.tools.extend(cleaned_tools)
    
    logger.info(f"MCP Initialization complete. Dynamic routes mapped for: {list(mcp_registry.functions.keys())}")

async def shutdown_mcp_infrastructure():
    """Clean execution teardown to prevent orphan operating system processes."""
    logger.info("Terminating background infrastructure processes...")
    await _exit_stack.aclose()