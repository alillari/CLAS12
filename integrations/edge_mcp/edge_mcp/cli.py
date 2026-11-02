"""Human-usable JSON CLI for the same operations exposed over MCP."""

import argparse
import json
from pathlib import Path
import sys

from .client import EdgeClient
from .models import HOSTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("hosts")
    p = commands.add_parser("inspect")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("--fast", action="store_true")
    for action in ["preview", "prepare"]:
        p = commands.add_parser(action)
        p.add_argument("host", choices=HOSTS)
        p.add_argument("--spec", required=True, type=Path, help="RunSpec JSON file")
        p.add_argument("--source", help="Local source directory to snapshot")
    for action in ["launch", "status", "stop", "logs", "artifact"]:
        p = commands.add_parser(action)
        p.add_argument("host", choices=HOSTS)
        p.add_argument("job_id")
        if action == "artifact":
            p.add_argument("path")
        if action == "logs":
            p.add_argument("--tail", type=int, default=100)
            p.add_argument("--source", choices=["stdout", "docker"], default="stdout")
    p = commands.add_parser("list")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("--limit", type=int, default=20)
    p = commands.add_parser("files")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("path")
    p.add_argument("--pattern", default="*")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--include-hidden", action="store_true")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--offset", type=int, default=0)
    p = commands.add_parser("read")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("path")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-bytes", type=int, default=65536)
    p = commands.add_parser("search")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("path")
    p.add_argument("query")
    p.add_argument("--pattern", default="*")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--include-hidden", action="store_true")
    p = commands.add_parser("images")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("--limit", type=int, default=50)
    p = commands.add_parser("image")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("image")
    p = commands.add_parser("test")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("--spec", required=True, type=Path)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--source")
    source.add_argument("--remote-source")
    p.add_argument("--wait", type=int, default=10)
    p = commands.add_parser("build-env")
    p.add_argument("host", choices=HOSTS)
    p.add_argument("environment_id")
    p.add_argument("--instructions", required=True, type=Path)
    p.add_argument("--base-image", default="pp_collision:alessio-latest")
    p.add_argument("--network", choices=["default", "none"], default="default")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--source")
    source.add_argument("--remote-context")
    for action in ["env-status", "env-stop"]:
        p = commands.add_parser(action)
        p.add_argument("host", choices=HOSTS)
        p.add_argument("environment_id")
    args = parser.parse_args()
    try:
        with EdgeClient() as client:
            if args.action == "hosts":
                result = client.list_hosts()
            elif args.action == "inspect":
                result = client.inspect_host(args.host, not args.fast)
            elif args.action in ["preview", "prepare"]:
                result = client.prepare_run(args.host, json.loads(args.spec.read_text()), args.source, args.action == "preview")
            elif args.action == "logs":
                result = client.read_run_logs(args.host, args.job_id, args.tail, args.source)
            elif args.action == "artifact":
                result = client.read_run_artifact(args.host, args.job_id, args.path)
            elif args.action == "list":
                result = client.list_runs(args.host, args.limit)
            elif args.action == "files":
                result = client.list_remote_files(args.host, args.path, args.pattern, args.recursive, args.include_hidden, args.limit, args.offset)
            elif args.action == "read":
                result = client.read_remote_file(args.host, args.path, args.offset, args.max_bytes)
            elif args.action == "search":
                result = client.search_remote_files(args.host, args.path, args.query, args.pattern, args.limit, args.include_hidden)
            elif args.action == "images":
                result = client.list_images(args.host, args.limit)
            elif args.action == "image":
                result = client.inspect_image(args.host, args.image)
            elif args.action == "test":
                result = client.run_tests(args.host, json.loads(args.spec.read_text()), args.source, args.remote_source, args.wait)
            elif args.action == "build-env":
                result = client.build_environment(args.host, args.environment_id, args.instructions.read_text(), args.base_image,
                                                  args.source, args.remote_context, args.network)
            elif args.action in ["env-status", "env-stop"]:
                method = client.get_environment_build if args.action == "env-status" else client.stop_environment_build
                result = method(args.host, args.environment_id)
            else:
                method = {"launch": client.launch_run, "status": client.get_run_status, "stop": client.stop_run}[args.action]
                result = method(args.host, args.job_id)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
