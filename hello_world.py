#!/usr/bin/env python3
"""Hello world script for testing resource consumption."""
import time

def main():
    start = time.perf_counter()
    print("Hello, World!")
    elapsed = time.perf_counter() - start
    print(f"Time taken: {elapsed:.6f} seconds")

if __name__ == "__main__":
    main()
