#!/usr/bin/env python3

#  Copyright OpenSearch Contributors
#  SPDX-License-Identifier: Apache-2.0

"""
E2E Test Runner for OpenSearch k-NN Plugin
"""
import os
import sys
import subprocess
import argparse

def run_tests(test_pattern=None, verbose=False):
    """Run e2e tests against external OpenSearch cluster"""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_pattern:
        cmd.extend(["-k", test_pattern])
    
    # Set environment
    env = os.environ.copy()
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run k-NN e2e tests")
    parser.add_argument("-k", "--pattern", help="Test pattern to match")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()

    exit_code = run_tests(args.pattern, args.verbose)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()