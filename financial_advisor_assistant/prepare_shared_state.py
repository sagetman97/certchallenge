"""
Prepare Shared State for Advanced Retriever Evaluation

This script runs the expensive steps once and saves the results
so they can be reused by multiple retriever evaluation scripts.
"""

import sys
import os
import pickle
from getpass import getpass

# Add project paths
project_root = "/mnt/c/AIProjects/AIE7-s09-10/financial_advisor_assistant"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all required packages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import pandas as pd
from datetime import datetime

# RAGAS imports
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import settings
from config.settings import settings

# LangSmith setup - ensure proper tracing
import os
from getpass import getpass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if LangSmith API key is already set
if os.environ.get("LANGSMITH_API_KEY"):
    print("✅ LangSmith API key found in environment")
else:
    print("⚠️ LangSmith API key not found in environment")
    print("   LangSmith tracing will be disabled")
    print("   You can set it with: export LANGSMITH_API_KEY=your_key_here")

# Set LangSmith project and tracing
os.environ["LANGCHAIN_PROJECT"] = "financial_advisor_shared_state_preparation"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Verify LangSmith setup
if os.environ.get("LANGSMITH_API_KEY"):
    try:
        from langsmith import Client
        client = Client()
        print("✅ LangSmith client created successfully")
        print("✅ Traces will be sent to LangSmith!")
    except Exception as e:
        print(f"❌ Error connecting to LangSmith: {e}")
        print("⚠️ LangSmith tracing will be disabled")
else:
    print("⚠️ LangSmith not configured - evaluation will run without tracing")

def step_1_setup_environment():
    """
    Step 1: Set up environment and API keys
    """
    print("🔧 **Step 1: Setting up environment**")
    print("=" * 50)
    
    # Set API key (use existing environment variable if available)
    if "OPENAI_API_KEY" not in os.environ:
        api_key = getpass("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print("✅ Using existing OpenAI API key from environment")
    
    print("✅ Environment set up successfully!")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print()

def step_2_load_documents():
    """
    Step 2: Load RAG Documents
    """
    print("📚 **Step 2: Loading RAG Documents**")
    print("=" * 50)
    
    # Load documents (following original notebook pattern)
    path = "../RAG Documents/"
    
    # Load text files first (our primary documents)
    loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} text documents")
    
    # Also load PDFs if available (following original notebook pattern)
    try:
        # Try different PDF loaders
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            pdf_loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader, show_progress=True)
            pdf_docs = pdf_loader.load()
            docs.extend(pdf_docs)
            print(f"✅ Loaded {len(pdf_docs)} PDF documents using PyMuPDF")
        except ImportError:
            print("⚠️ PyMuPDF not available, trying alternative PDF loader")
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                pdf_loader = DirectoryLoader(path, glob="*.pdf", loader_cls=UnstructuredPDFLoader, show_progress=True)
                pdf_docs = pdf_loader.load()
                docs.extend(pdf_docs)
                print(f"✅ Loaded {len(pdf_docs)} PDF documents using UnstructuredPDF")
            except ImportError:
                print("⚠️ UnstructuredPDF not available, skipping PDFs")
                print("💡 To load PDFs, install: pip install pymupdf or pip install unstructured[pdf]")
    except Exception as e:
        print(f"⚠️ PDF loading failed: {e}")
        print("💡 To fix PDF loading, install: pip install pymupdf unstructured[pdf]")
    
    print(f"📊 Total documents loaded: {len(docs)}")
    print()
    
    return docs

def step_3_create_golden_dataset(docs):
    """
    Step 3: Create Golden Dataset with RAGAS
    """
    print("🔧 **Step 3: Creating Golden Dataset with RAGAS**")
    print("=" * 60)
    
    # Initialize generator LLM and embeddings (following original notebook exactly)
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4.1")
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings()
    )
    
    # Create TestsetGenerator
    generator = TestsetGenerator(
        llm=generator_llm, 
        embedding_model=generator_embeddings
    )
    
    print("🔄 Generating synthetic test dataset...")
    print("   This will create knowledge graph and generate complex queries")
    print("   Note: 'Property summary already exists' warnings are normal - RAGAS is building a knowledge graph")
    
    # Generate dataset from our documents (with error handling)
    try:
        dataset = generator.generate_with_langchain_docs(
            docs[:10],  # Use first 10 documents for generation
            testset_size=28  # Generate 28 test cases
        )
    except Exception as e:
        print(f"⚠️ RAGAS generation failed: {e}")
        print("🔄 Trying with even smaller dataset...")
        try:
            dataset = generator.generate_with_langchain_docs(
                docs[:5],  # Use first 5 documents for generation
                testset_size=28  # Generate 28 test cases
            )
        except Exception as e2:
            print(f"❌ RAGAS generation failed again: {e2}")
            print("🔄 Creating manual test dataset as fallback...")
            # Create a simple manual dataset as fallback
            from ragas import Testset
            manual_data = [
                {
                    "question": "What is the difference between term and whole life insurance?",
                    "context": "Term life insurance provides temporary coverage while whole life offers permanent protection.",
                    "answer": "Term life insurance provides coverage for a specific period with lower premiums, while whole life insurance offers permanent coverage with cash value accumulation."
                },
                {
                    "question": "How do I calculate my life insurance needs?",
                    "context": "Multiple methods exist for calculating life insurance coverage needs.",
                    "answer": "Calculate life insurance needs using: Human Life Value, Needs-Based method, DIME method, or Rule of Thumb (10-15x annual income)."
                },
                {
                    "question": "What are the benefits of whole life insurance?",
                    "context": "Whole life insurance offers permanent coverage with cash value benefits.",
                    "answer": "Whole life insurance benefits include: permanent coverage, guaranteed premiums, cash value accumulation, tax-deferred growth, and potential dividends."
                }
            ]
            dataset = Testset.from_dict(manual_data)
            print("✅ Created manual test dataset as fallback")
    
    print(f"✅ Generated {len(dataset.samples)} test cases")
    print()
    
    # Show sample questions
    print("📝 Sample Questions from Generated Dataset:")
    for i, sample in enumerate(dataset.samples[:5], 1):
        print(f"  {i}. {sample.eval_sample.user_input[:80]}...")
    
    print()
    return dataset

def step_4_create_vector_store(docs):
    """
    Step 4: Create Vector Store (Shared Resource)
    """
    print("🗄️ **Step 4: Creating Vector Store**")
    print("=" * 50)
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    split_documents = text_splitter.split_documents(docs)
    print(f"📄 Split into {len(split_documents)} chunks")
    
    # Create vector store
    embeddings = OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)
    vector_store = Qdrant.from_documents(
        split_documents,
        embeddings,
        collection_name="financial_advisor_eval"
    )
    
    print("✅ Vector store created successfully")
    print()
    
    return vector_store, split_documents

def save_shared_state(docs, dataset, vector_store, split_documents):
    """
    Save all shared state to files
    """
    print("💾 **Saving Shared State**")
    print("=" * 50)
    
    # Create shared_state directory
    os.makedirs("shared_state", exist_ok=True)
    
    # Save documents
    with open("shared_state/documents.pkl", "wb") as f:
        pickle.dump(docs, f)
    print("💾 Documents saved: shared_state/documents.pkl")
    
    # Save split documents
    with open("shared_state/split_documents.pkl", "wb") as f:
        pickle.dump(split_documents, f)
    print("💾 Split documents saved: shared_state/split_documents.pkl")
    
    # Save golden dataset
    dataset.to_pandas().to_csv("shared_state/golden_dataset.csv", index=False)
    print("💾 Golden dataset saved: shared_state/golden_dataset.csv")
    
    # Save vector store info (we'll recreate it in each script)
    vector_store_info = {
        "collection_name": "financial_advisor_eval",
        "embedding_model": settings.OPENAI_EMBEDDING_MODEL
    }
    with open("shared_state/vector_store_info.pkl", "wb") as f:
        pickle.dump(vector_store_info, f)
    print("💾 Vector store info saved: shared_state/vector_store_info.pkl")
    
    print("✅ All shared state saved!")
    print()

def main():
    """
    Main function - prepare shared state
    """
    print("🚀 **Preparing Shared State for Advanced Retriever Evaluation**")
    print("=" * 70)
    print("This will run the expensive steps once and save the results")
    print("for reuse by multiple retriever evaluation scripts.")
    print()
    
    try:
        # Step 1: Setup
        step_1_setup_environment()
        
        # Step 2: Load documents
        docs = step_2_load_documents()
        
        # Step 3: Create golden dataset
        dataset = step_3_create_golden_dataset(docs)
        
        # Step 4: Create vector store
        vector_store, split_documents = step_4_create_vector_store(docs)
        
        # Save shared state
        save_shared_state(docs, dataset, vector_store, split_documents)
        
        print("✅ **Shared State Preparation Complete!**")
        print("📁 Files saved in shared_state/ directory")
        print("🚀 Ready to run individual retriever evaluation scripts!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 