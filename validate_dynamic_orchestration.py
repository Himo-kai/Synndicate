#!/usr/bin/env python3
"""
Dynamic Orchestration Validation Script

This script validates the dynamic orchestration system with real workloads
and measures performance metrics.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Set environment variables
os.environ["SYN_SEED"] = "1337"

async def main():
    """Main validation function."""
    print("🚀 Starting Dynamic Orchestration Validation")
    print("=" * 60)
    
    try:
        # Import after path setup
        from synndicate.core.enhanced_orchestrator import EnhancedOrchestrator
        from synndicate.models.manager import ModelManager
        
        # Initialize components
        print("📋 Initializing components...")
        
        # Setup model manager
        model_manager = ModelManager()
        await model_manager.initialize()
        print("✅ Model manager initialized")
        
        # Setup enhanced orchestrator
        orchestrator = EnhancedOrchestrator(
            max_agents=5,
            idle_timeout=120.0,
            performance_threshold=0.6
        )
        print("✅ Enhanced orchestrator initialized")
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Simple Coding Task",
                "query": "Implement a Python function to calculate fibonacci numbers",
                "workflow": "dynamic",
                "expected_agents": ["planner", "coder"]
            },
            {
                "name": "API Development Task", 
                "query": "Create a FastAPI endpoint for user registration with validation",
                "workflow": "dynamic",
                "context": {"framework": "FastAPI", "validation": "pydantic"},
                "expected_agents": ["planner", "coder"]
            },
            {
                "name": "Architecture Planning",
                "query": "Design a microservices architecture for an e-commerce platform",
                "workflow": "dynamic", 
                "context": {"domain": "e-commerce", "scale": "high"},
                "expected_agents": ["planner"]
            },
            {
                "name": "Auto Workflow Selection",
                "query": "What are the best practices for Python error handling?",
                "workflow": "auto",
                "expected_agents": ["planner"]
            }
        ]
        
        results = []
        total_start_time = time.time()
        
        print(f"\n🧪 Running {len(test_scenarios)} test scenarios...")
        print("-" * 60)
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n[{i}/{len(test_scenarios)}] Testing: {scenario['name']}")
            
            start_time = time.time()
            
            try:
                # Execute the query
                result = await orchestrator.process_query(
                    scenario["query"],
                    context=scenario.get("context"),
                    workflow=scenario["workflow"]
                )
                
                execution_time = time.time() - start_time
                
                # Analyze result
                success = result.success if result else False
                response_length = len(result.response.response) if result and result.response else 0
                confidence = result.response.confidence if result and result.response else 0
                workflow_type = result.workflow_type if result else "unknown"
                
                # Get orchestration status
                status = orchestrator.get_orchestration_status()
                dynamic_stats = status.get("dynamic_orchestration", {})
                
                scenario_result = {
                    "scenario": scenario["name"],
                    "success": success,
                    "execution_time": execution_time,
                    "workflow_type": workflow_type,
                    "response_length": response_length,
                    "confidence": confidence,
                    "agent_pool_size": status.get("agent_pool_size", 0),
                    "active_agents": dynamic_stats.get("active_agents", 0),
                    "total_recruitments": status.get("recruitment_history", 0)
                }
                
                results.append(scenario_result)
                
                # Print results
                status_icon = "✅" if success else "❌"
                print(f"   {status_icon} Status: {'SUCCESS' if success else 'FAILED'}")
                print(f"   ⏱️  Execution Time: {execution_time:.2f}s")
                print(f"   🔧 Workflow Type: {workflow_type}")
                print(f"   📊 Response Length: {response_length} chars")
                print(f"   🎯 Confidence: {confidence:.2f}")
                print(f"   👥 Active Agents: {dynamic_stats.get('active_agents', 0)}")
                
                if not success:
                    print(f"   ⚠️  Error: {getattr(result, 'error', 'Unknown error')}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"   ❌ CRASHED: {e}")
                
                results.append({
                    "scenario": scenario["name"],
                    "success": False,
                    "execution_time": execution_time,
                    "error": str(e)
                })
        
        total_time = time.time() - total_start_time
        
        # Calculate summary statistics
        successful_tests = sum(1 for r in results if r["success"])
        success_rate = (successful_tests / len(results)) * 100
        avg_execution_time = sum(r["execution_time"] for r in results) / len(results)
        avg_confidence = sum(r.get("confidence", 0) for r in results if r.get("confidence", 0) > 0)
        avg_confidence = avg_confidence / max(1, sum(1 for r in results if r.get("confidence", 0) > 0))
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {len(results)}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {len(results) - successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Execution Time: {avg_execution_time:.2f}s")
        print(f"Average Confidence: {avg_confidence:.2f}")
        
        # Get final orchestration status
        final_status = orchestrator.get_orchestration_status()
        dynamic_stats = final_status.get("dynamic_orchestration", {})
        
        print(f"\n🔧 ORCHESTRATION METRICS")
        print("-" * 30)
        print(f"Agent Pool Size: {final_status.get('agent_pool_size', 0)}")
        print(f"Total Recruitments: {final_status.get('recruitment_history', 0)}")
        print(f"Total Dismissals: {final_status.get('dismissal_history', 0)}")
        print(f"Active Agents: {dynamic_stats.get('active_agents', 0)}")
        print(f"Idle Agents: {dynamic_stats.get('idle_agents', 0)}")
        
        # Save detailed results
        results_file = project_root / "dynamic_orchestration_validation_results.json"
        detailed_results = {
            "summary": {
                "total_tests": len(results),
                "successful": successful_tests,
                "failed": len(results) - successful_tests,
                "success_rate": success_rate,
                "total_time": total_time,
                "avg_execution_time": avg_execution_time,
                "avg_confidence": avg_confidence
            },
            "orchestration_metrics": {
                "agent_pool_size": final_status.get("agent_pool_size", 0),
                "total_recruitments": final_status.get("recruitment_history", 0),
                "total_dismissals": final_status.get("dismissal_history", 0),
                "dynamic_stats": dynamic_stats
            },
            "test_results": results
        }
        
        with open(results_file, "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Final assessment
        if success_rate >= 80:
            print("\n🎉 VALIDATION PASSED!")
            print("Dynamic orchestration system is performing well and ready for production use.")
        elif success_rate >= 60:
            print("\n⚠️  VALIDATION PARTIAL SUCCESS")
            print("Dynamic orchestration system is functional but may need optimization.")
        else:
            print("\n❌ VALIDATION FAILED")
            print("Dynamic orchestration system needs significant improvements.")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        await orchestrator.dynamic_orchestrator.cleanup_idle_agents()
        print("✅ Cleanup complete")
        
    except Exception as e:
        print(f"\n💥 VALIDATION CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success_rate >= 80


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
