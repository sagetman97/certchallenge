"""
Simple test script to debug LangGraph HumanMessage validation.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config.settings import settings
from qdrant_client import QdrantClient
from agents.teams.tools import initialize_vector_store

def test_simple_langgraph():
    """Test basic LangGraph functionality."""
    
    # Initialize components
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    initialize_vector_store(client)
    llm = ChatOpenAI(model=settings.OPENAI_MODEL)
    
    # Test simple message creation
    try:
        message = HumanMessage(content="What is whole life insurance?")
        print("✅ HumanMessage created successfully")
        print(f"Message content: {message.content}")
        print(f"Message type: {type(message)}")
    except Exception as e:
        print(f"❌ HumanMessage creation failed: {e}")
        return
    
    # Test basic LLM call
    try:
        response = llm.invoke([message])
        print("✅ LLM call successful")
        print(f"Response: {response.content[:100]}...")
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return
    
    print("✅ Basic LangGraph test completed successfully")

if __name__ == "__main__":
    test_simple_langgraph() 