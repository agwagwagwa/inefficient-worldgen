"""
Main entry point for the package.

Usage:
    uv run python -m inefficient_worldgen <command> [args]

Commands:
    extract  - Extract chunks from a Minecraft world
    train    - Train the diffusion model
    generate - Generate a world from a trained model
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    # Remove the command from argv so argparse in submodules works correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "extract":
        from .extract import main as extract_main

        extract_main()
    elif command == "train":
        from .train import main as train_main

        train_main()
    elif command == "generate":
        from .generate import main as generate_main

        generate_main()
    elif command == "serve":
        from .api import run_server
        import argparse

        parser = argparse.ArgumentParser(description="Run the API server")
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args()
        run_server(host=args.host, port=args.port)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
