"""
Test LangSmith Tracing

This script tests if LangSmith tracing is working properly.
"""

import os
import sys
from getpass import getpass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project paths
project_root = "/mnt/c/AIProjects/AIE7-s09-10/financial_advisor_assistant"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# LangSmith setup
print("🔧 **LangSmith Setup Test**")
print("=" * 50)

# Check if LangSmith API key is already set
if os.environ.get("LANGSMITH_API_KEY"):
    print("✅ LangSmith API key found in environment")
    print(f"   Key: {os.environ.get('LANGSMITH_API_KEY')[:10]}...")
else:
    print("⚠️ LangSmith API key not found in environment")
    print("   You can set it with: export LANGSMITH_API_KEY=your_key_here")
    print("   Or add it to your .env file")
    exit(0)

# Set LangSmith project and tracing
os.environ["LANGCHAIN_PROJECT"] = "langsmith_test"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Test LangSmith connection
try:
    from langsmith import Client
    client = Client()
    print("✅ LangSmith client created successfully")
    
    # Test a simple LLM call
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm.invoke("Say 'Hello, LangSmith tracing is working!'")
    
    print("✅ LangSmith tracing test successful!")
    print(f"Response: {response.content}")
    print()
    print("📊 Check LangSmith for traces:")
    print("   - Go to https://smith.langchain.com")
    print("   - Look for project: langsmith_test")
    print("   - You should see traces for this test")
    
except Exception as e:
    print(f"❌ LangSmith test failed: {e}")
    print("Common issues:")
    print("   - Invalid API key")
    print("   - Network connectivity issues")
    print("   - LangSmith service down") 