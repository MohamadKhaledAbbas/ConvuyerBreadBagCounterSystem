#!/usr/bin/env python3
"""
Run the FastAPI endpoint server.

Usage:
    python run_endpoint.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_endpoint.py                    # Default: 0.0.0.0:8000
    python run_endpoint.py --port 8080        # Custom port
    python run_endpoint.py --reload           # Development mode with auto-reload
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run the Conveyor Counter API server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')

    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install uvicorn")
        uvicorn = None
        sys.exit(1)

    print("=" * 60)
    print("Conveyor Bread Bag Counter - API Server")
    print("=" * 60)
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print("=" * 60)

    uvicorn.run(
        "src.endpoint.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == '__main__':
    main()
