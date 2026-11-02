"""Check the actual registered MCP command from an unrelated working directory."""

import asyncio
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    codex = "/usr/lib/chatgpt/resources/codex"
    config = json.loads(subprocess.check_output([codex, "mcp", "get", "clas12-edge", "--json"], text=True))
    transport = config["transport"]
    assert config["enabled"] and transport["type"] == "stdio"
    params = StdioServerParameters(command=transport["command"], args=transport.get("args", []), cwd="/tmp")
    report = {"configuration": config, "hosts": {}}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=60)) as session:
            initialized = await session.initialize()
            report["server"] = initialized.serverInfo.model_dump()
            report["tools"] = [t.name for t in (await session.list_tools()).tools]
            assert len(report["tools"]) >= 10
            for host in ["a02c", "a518"]:
                result = await session.call_tool("inspect_host", {"host": host, "details": False})
                assert not result.isError
                data = result.structuredContent
                assert data and not data["errors"]
                runs = await session.call_tool("list_runs", {"host": host, "limit": 100})
                assert not runs.isError
                active = [r["job_id"] for r in runs.structuredContent["runs"] if r["status"] in ["created", "running", "paused", "restarting"]]
                report["hosts"][host] = {"hostname": data["hostname"], "managed_jobs": len(runs.structuredContent["runs"]), "active_managed_jobs": active}
    report["passed"] = True
    target = Path(sys.argv[1])
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"configured_server": config["name"], "passed": True, "tools": len(report["tools"]), "hosts": report["hosts"]}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
