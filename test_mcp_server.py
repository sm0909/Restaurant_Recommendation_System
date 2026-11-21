"""Quick check of the MCP server: starts configure_tools_data_mcp_server.py over stdio and calls one tool.

Needs structured-restaurant-data.json next to the server: python3 evaluation/build_knowledge_base.py
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = str(Path(__file__).parent / "configure_tools_data_mcp_server.py")


async def run_test():
    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_restaurant_info", arguments={"restaurant_name": "Iron"})
            print("\n--- START SCREENSHOT ---")
            print(result.content[0].text)
            print("--- END SCREENSHOT ---\n")


if __name__ == "__main__":
    asyncio.run(run_test())
