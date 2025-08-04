"""
Comprehensive System Evaluation for Financial Advisor Assistant

This script evaluates the entire system including:
- Advanced retrieval techniques
- Multi-agent system performance
- RAGAS-based evaluation
- Synthetic data generation
"""

import os
import sys
from typing import Dict, Any, List
import pandas as pd
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config.settings import get_settings
from agents.rag_agent import EnhancedRAGAgent
from agents.orchestrator import MultiAgentOrchestrator
from evaluation.synthetic_data import SyntheticDataGenerator
from evaluation.evaluator import RAGEvaluator
from retrieval import (
    HybridRetriever,
    EnhancedMultiQueryRetriever,
    EnhancedParentDocumentRetriever,
    EnhancedContextualCompressionRetriever,
    EnhancedEnsembleRetriever
)

settings = get_settings()

class SystemEvaluator:
    """
    Comprehensive system evaluator for the Financial Advisor Assistant.
    """
    
    def __init__(self):
        """Initialize the system evaluator."""
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL
        )
        
        # Initialize components
        self.rag_agent = EnhancedRAGAgent(retrieval_method="ensemble")
        self.orchestrator = MultiAgentOrchestrator(
            llm=self.llm,
            rag_agent=self.rag_agent,
            tavily_api_key=settings.TAVILY_API_KEY
        )
        
        # Initialize evaluation components
        self.synthetic_generator = SyntheticDataGenerator(
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        self.evaluator = RAGEvaluator(
            llm=self.llm,
            embeddings=self.embeddings,
            rag_chain=self._create_rag_chain_for_evaluation()
        )
    
    def _create_rag_chain_for_evaluation(self):
        """
        Create RAG chain for evaluation purposes.
        
        Returns:
            RAG chain function
        """
        def rag_chain(question: str) -> str:
            try:
                result = self.rag_agent.query(question)
                return result.get("response", "No response generated")
            except Exception as e:
                return f"Error: {str(e)}"
        
        return rag_chain
    
    def evaluate_retrieval_methods(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate different retrieval methods.
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Evaluation results for each retrieval method
        """
        print("🔍 Evaluating Retrieval Methods...")
        
        retrieval_methods = {
            "basic": "Basic similarity search",
            "hybrid": "Hybrid BM25 + semantic search",
            "multi_query": "Multi-query retrieval",
            "parent_document": "Parent-document retrieval",
            "ensemble": "Ensemble retrieval"
        }
        
        results = {}
        
        for method_name, description in retrieval_methods.items():
            print(f"  Testing {method_name}: {description}")
            
            try:
                # Set retrieval method
                self.rag_agent.set_retrieval_method(method_name)
                
                # Test queries
                method_results = []
                for query in test_queries:
                    result = self.rag_agent.query(query)
                    method_results.append({
                        "query": query,
                        "response_length": len(result.get("response", "")),
                        "confidence_score": result.get("confidence_score", 0),
                        "retrieval_method": method_name
                    })
                
                # Calculate average metrics
                avg_confidence = sum(r["confidence_score"] for r in method_results) / len(method_results)
                avg_response_length = sum(r["response_length"] for r in method_results) / len(method_results)
                
                results[method_name] = {
                    "description": description,
                    "average_confidence": avg_confidence,
                    "average_response_length": avg_response_length,
                    "query_results": method_results
                }
                
            except Exception as e:
                results[method_name] = {
                    "description": description,
                    "error": str(e)
                }
        
        return results
    
    def evaluate_multi_agent_system(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate the multi-agent system performance.
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Multi-agent evaluation results
        """
        print("🤖 Evaluating Multi-Agent System...")
        
        results = {
            "total_queries": len(test_queries),
            "agent_usage": {},
            "routing_accuracy": {},
            "response_quality": []
        }
        
        for i, query in enumerate(test_queries):
            print(f"  Processing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            try:
                # Process through orchestrator
                result = self.orchestrator.process_query(query)
                
                # Extract metrics
                agent_type = result.get("metadata", {}).get("agent_type", "UNKNOWN")
                confidence = result.get("metadata", {}).get("confidence", 0)
                response_length = len(result.get("response", ""))
                
                # Track agent usage
                if agent_type not in results["agent_usage"]:
                    results["agent_usage"][agent_type] = 0
                results["agent_usage"][agent_type] += 1
                
                # Track response quality
                results["response_quality"].append({
                    "query": query,
                    "agent_type": agent_type,
                    "confidence": confidence,
                    "response_length": response_length
                })
                
            except Exception as e:
                print(f"    Error processing query: {e}")
                results["response_quality"].append({
                    "query": query,
                    "error": str(e)
                })
        
        return results
    
    def generate_and_evaluate_synthetic_data(self) -> Dict[str, Any]:
        """
        Generate synthetic test data and evaluate system performance.
        
        Returns:
            Evaluation results with synthetic data
        """
        print("🧪 Generating and Evaluating Synthetic Data...")
        
        try:
            # Generate synthetic test data
            print("  Generating synthetic test data...")
            testset = self.synthetic_generator.generate_domain_specific_tests("life_insurance")
            
            # Save testset
            self.synthetic_generator.save_testset(testset, "life_insurance_evaluation")
            
            # Evaluate with RAGAS
            print("  Running RAGAS evaluation...")
            evaluation_results = self.evaluator.evaluate_rag_chain(
                testset, 
                "life_insurance_ragas_evaluation"
            )
            
            # Generate evaluation report
            self.evaluator.generate_evaluation_report(
                evaluation_results,
                "evaluation_reports/life_insurance_evaluation_report.md"
            )
            
            return {
                "testset_size": len(testset),
                "evaluation_results": evaluation_results,
                "testset_columns": list(testset.columns)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive system evaluation.
        
        Returns:
            Comprehensive evaluation results
        """
        print("🚀 Starting Comprehensive System Evaluation")
        print("=" * 60)
        
        # Test queries covering different aspects
        test_queries = [
            "What is whole life insurance and how does it work?",
            "How much life insurance coverage do I need if I make $75,000 per year?",
            "What are the differences between term and whole life insurance?",
            "How does life insurance fit into a financial portfolio?",
            "What are the current trends in life insurance rates?",
            "How do I calculate life insurance needs for a family?",
            "What are the benefits of universal life insurance?",
            "How does life insurance work for business owners?",
            "What factors affect life insurance premiums?",
            "How does life insurance fit into retirement planning?"
        ]
        
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_queries": test_queries,
            "retrieval_evaluation": {},
            "multi_agent_evaluation": {},
            "synthetic_data_evaluation": {},
            "system_summary": {}
        }
        
        # 1. Evaluate retrieval methods
        print("\n📊 Phase 1: Retrieval Method Evaluation")
        results["retrieval_evaluation"] = self.evaluate_retrieval_methods(test_queries)
        
        # 2. Evaluate multi-agent system
        print("\n🤖 Phase 2: Multi-Agent System Evaluation")
        results["multi_agent_evaluation"] = self.evaluate_multi_agent_system(test_queries)
        
        # 3. Generate and evaluate synthetic data
        print("\n🧪 Phase 3: Synthetic Data Generation and Evaluation")
        results["synthetic_data_evaluation"] = self.generate_and_evaluate_synthetic_data()
        
        # 4. Generate system summary
        print("\n📋 Phase 4: Generating System Summary")
        results["system_summary"] = self._generate_system_summary(results)
        
        # Save comprehensive results
        self._save_evaluation_results(results)
        
        print("\n✅ Comprehensive Evaluation Complete!")
        print("=" * 60)
        
        return results
    
    def _generate_system_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate system performance summary.
        
        Args:
            results: Evaluation results
            
        Returns:
            System summary
        """
        summary = {
            "overall_performance": "Good",
            "recommendations": [],
            "key_metrics": {}
        }
        
        # Analyze retrieval performance
        retrieval_results = results.get("retrieval_evaluation", {})
        if retrieval_results:
            best_method = max(
                retrieval_results.items(),
                key=lambda x: x[1].get("average_confidence", 0) if "error" not in x[1] else 0
            )
            summary["key_metrics"]["best_retrieval_method"] = best_method[0]
            summary["key_metrics"]["best_confidence_score"] = best_method[1].get("average_confidence", 0)
        
        # Analyze multi-agent performance
        multi_agent_results = results.get("multi_agent_evaluation", {})
        if multi_agent_results:
            agent_usage = multi_agent_results.get("agent_usage", {})
            if agent_usage:
                most_used_agent = max(agent_usage.items(), key=lambda x: x[1])
                summary["key_metrics"]["most_used_agent"] = most_used_agent[0]
                summary["key_metrics"]["agent_usage_distribution"] = agent_usage
        
        # Analyze synthetic data evaluation
        synthetic_results = results.get("synthetic_data_evaluation", {})
        if "evaluation_results" in synthetic_results:
            eval_results = synthetic_results["evaluation_results"]
            overall_score = eval_results.get("overall_score", 0)
            
            if overall_score > 0.8:
                summary["overall_performance"] = "Excellent"
            elif overall_score > 0.6:
                summary["overall_performance"] = "Good"
            else:
                summary["overall_performance"] = "Needs Improvement"
            
            summary["key_metrics"]["ragas_overall_score"] = overall_score
            summary["key_metrics"]["ragas_retrieval_score"] = eval_results.get("retrieval_score", 0)
            summary["key_metrics"]["ragas_generation_score"] = eval_results.get("generation_score", 0)
        
        # Generate recommendations
        if summary["key_metrics"].get("ragas_overall_score", 0) < 0.7:
            summary["recommendations"].append("Consider improving retrieval accuracy")
        
        if not multi_agent_results.get("agent_usage", {}):
            summary["recommendations"].append("Multi-agent system needs more testing")
        
        return summary
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
        """
        try:
            # Create evaluation directory
            os.makedirs("evaluation_results", exist_ok=True)
            
            # Save comprehensive results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"evaluation_results/comprehensive_evaluation_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"📁 Evaluation results saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving evaluation results: {e}")


def main():
    """Main evaluation function."""
    print("🎯 Financial Advisor Assistant - Comprehensive System Evaluation")
    print("=" * 70)
    
    # Check environment
    try:
        settings.validate()
        print("✅ Environment variables validated")
    except ValueError as e:
        print(f"❌ Environment error: {e}")
        print("Please set up your .env file with required API keys")
        return
    
    # Initialize evaluator
    evaluator = SystemEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Print summary
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 40)
    
    summary = results.get("system_summary", {})
    print(f"Overall Performance: {summary.get('overall_performance', 'Unknown')}")
    
    key_metrics = summary.get("key_metrics", {})
    if key_metrics:
        print("\nKey Metrics:")
        for metric, value in key_metrics.items():
            print(f"  {metric}: {value}")
    
    recommendations = summary.get("recommendations", [])
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print(f"\n📁 Detailed results saved to: evaluation_results/")
    print("📄 Evaluation report generated: evaluation_reports/")


if __name__ == "__main__":
    main() 