#!/usr/bin/env python3
"""
Simple Dynamic Orchestration Validation Runner

This script runs the comprehensive test suite and provides validation results.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run dynamic orchestration validation."""
    print("🚀 Dynamic Orchestration System Validation")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Activate virtual environment and run tests
    print("📋 Running comprehensive test suite...")
    
    try:
        # Run the dynamic orchestration tests
        result = subprocess.run([
            "bash", "-c", 
            "source venv/bin/activate && python -m pytest tests/test_dynamic_orchestration.py -v --tb=short"
        ], capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ VALIDATION PASSED!")
            print("Dynamic orchestration system tests completed successfully.")
        else:
            print(f"\n❌ VALIDATION FAILED (exit code: {result.returncode})")
            print("Some tests failed. Check output above for details.")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
