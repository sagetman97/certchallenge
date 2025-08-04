"""
Test script for the Multi-Agent Financial Advisor Assistant.

This script tests the various agents and demonstrates the system's capabilities.
"""

import os
import sys
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from config.settings import get_settings
from agents.rag_agent import RAGAgent
from agents.orchestrator import MultiAgentOrchestrator

def test_multi_agent_system():
    """
    Test the multi-agent system with various query types.
    """
    settings = get_settings()
    
    print("🚀 Testing Multi-Agent Financial Advisor Assistant")
    print("=" * 60)
    
    # Initialize components
    print("📋 Initializing components...")
    
    # Initialize RAG agent
    rag_agent = RAGAgent(
        llm=settings.OPENAI_MODEL,
        embeddings=settings.EMBEDDINGS_MODEL,
        qdrant_client=None  # We'll skip Qdrant for testing
    )
    
    # Initialize Multi-Agent Orchestrator
    orchestrator = MultiAgentOrchestrator(
        llm=settings.OPENAI_MODEL,
        rag_agent=rag_agent,
        tavily_api_key=settings.TAVILY_API_KEY
    )
    
    print("✅ Components initialized successfully")
    print()
    
    # Test queries
    test_queries = [
        {
            "query": "What is whole life insurance and how does it work?",
            "expected_agent": "RAG",
            "description": "Basic knowledge question"
        },
        {
            "query": "How much life insurance coverage do I need if I make $75,000 per year?",
            "expected_agent": "CALCULATOR",
            "description": "Calculator request"
        },
        {
            "query": "What are current term life insurance rates for a 35-year-old?",
            "expected_agent": "EXTERNAL_SEARCH",
            "description": "Current market information request"
        },
        {
            "query": "How does life insurance fit into a diversified portfolio?",
            "expected_agent": "RAG",
            "description": "Portfolio integration question"
        },
        {
            "query": "Research different types of universal life insurance policies",
            "expected_agent": "RESEARCH",
            "description": "Research request"
        }
    ]
    
    print("🧪 Testing Query Routing and Agent Responses")
    print("-" * 60)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Agent: {test_case['expected_agent']}")
        
        try:
            # Process query through orchestrator
            result = orchestrator.process_query(test_case['query'])
            
            # Extract results
            actual_agent = result.get("metadata", {}).get("agent_type", "UNKNOWN")
            response = result.get("response", "No response")
            confidence = result.get("metadata", {}).get("confidence", 0.0)
            agents_used = result.get("metadata", {}).get("agents_used", [])
            
            print(f"✅ Actual Agent: {actual_agent}")
            print(f"📊 Confidence: {confidence:.2f}")
            print(f"🔧 Agents Used: {', '.join(agents_used)}")
            print(f"💬 Response Preview: {response[:200]}...")
            
            # Check if routing was correct
            if actual_agent == test_case['expected_agent']:
                print("✅ Routing correct!")
            else:
                print(f"⚠️  Routing different than expected (got {actual_agent}, expected {test_case['expected_agent']})")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)
    
    # Test agent capabilities
    print("\n🔧 Testing Agent Capabilities")
    print("-" * 60)
    
    capabilities = orchestrator.get_agent_capabilities()
    for agent_name, capability in capabilities.items():
        print(f"\n📋 {agent_name} Agent:")
        print(f"   Description: {capability['description']}")
        print(f"   Capabilities: {', '.join(capability['capabilities'])}")
        print(f"   Best for: {capability['best_for']}")
    
    # Test conversation summary
    print("\n📊 Testing Conversation Summary")
    print("-" * 60)
    
    summary = orchestrator.get_conversation_summary()
    print(f"Total Queries: {summary.get('total_queries', 0)}")
    print(f"Agent Usage: {summary.get('agent_usage', {})}")
    print(f"Query Types: {summary.get('query_types', [])}")
    
    print("\n🎉 Multi-Agent System Test Complete!")
    print("=" * 60)

def test_confidence_assessment():
    """
    Test the confidence assessment functionality.
    """
    print("\n🧠 Testing Confidence Assessment")
    print("-" * 60)
    
    from utils.confidence import assess_confidence_score, should_use_external_search
    
    test_responses = [
        {
            "response": "Whole life insurance is a type of permanent life insurance that provides coverage for the entire lifetime of the insured person. It includes a cash value component that grows over time and can be accessed through loans or withdrawals.",
            "query": "What is whole life insurance?",
            "expected_confidence": "high"
        },
        {
            "response": "I don't have specific information about that topic.",
            "query": "What are current market rates?",
            "expected_confidence": "low"
        },
        {
            "response": "Life insurance can be calculated using several methods including the human life value method, needs-based analysis, and the DIME method. The specific amount depends on factors like income, debts, and family situation.",
            "query": "How do I calculate life insurance needs?",
            "expected_confidence": "medium"
        }
    ]
    
    for i, test_case in enumerate(test_responses, 1):
        print(f"\n📝 Confidence Test {i}")
        print(f"Query: {test_case['query']}")
        print(f"Response: {test_case['response'][:100]}...")
        
        confidence = assess_confidence_score(test_case['response'], test_case['query'], [])  # Empty list for retrieved_docs
        should_search = should_use_external_search(confidence)
        
        print(f"📊 Confidence Score: {confidence:.2f}")
        print(f"🔍 Should Use External Search: {should_search}")
        print(f"📈 Expected: {test_case['expected_confidence']}")

if __name__ == "__main__":
    # Check if environment variables are set
    settings = get_settings()
    
    try:
        settings.validate()
        print("✅ Environment variables validated")
    except ValueError as e:
        print(f"❌ Environment error: {e}")
        print("Please set up your .env file with required API keys")
        exit(1)
    
    # Run tests
    test_multi_agent_system()
    test_confidence_assessment() 