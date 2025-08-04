"""
Evaluate BM25 Retriever with RAGAS

This script loads the shared state and evaluates BM25 retrieval
using RAGAS metrics with LangSmith tracing.
"""

import sys
import os
import pickle
import json
from datetime import datetime

# Add project paths
project_root = "/mnt/c/AIProjects/AIE7-s09-10/financial_advisor_assistant"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all required packages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
import pandas as pd

# RAGAS imports
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper

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
os.environ["LANGCHAIN_PROJECT"] = "financial_advisor_bm25_evaluation"
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

def load_shared_state():
    """
    Load the shared state (documents, dataset, vector store)
    """
    print("📂 **Loading Shared State**")
    print("=" * 50)
    
    # Check if shared state exists
    if not os.path.exists("shared_state"):
        print("❌ Shared state not found. Please run prepare_shared_state.py first.")
        return None, None, None, None
    
    # Load documents
    with open("shared_state/documents.pkl", "rb") as f:
        docs = pickle.load(f)
    print("✅ Loaded documents")
    
    # Load split documents
    with open("shared_state/split_documents.pkl", "rb") as f:
        split_documents = pickle.load(f)
    print("✅ Loaded split documents")
    
    # Load golden dataset
    dataset_df = pd.read_csv("shared_state/golden_dataset.csv")
    
    # Fix the reference_contexts column - convert string representations back to lists
    import ast
    if 'reference_contexts' in dataset_df.columns:
        dataset_df['reference_contexts'] = dataset_df['reference_contexts'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    from ragas import EvaluationDataset
    dataset = EvaluationDataset.from_pandas(dataset_df)
    print("✅ Loaded golden dataset")
    
    # Load vector store info and recreate vector store
    with open("shared_state/vector_store_info.pkl", "rb") as f:
        vector_store_info = pickle.load(f)
    
    embeddings = OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)
    vector_store = Qdrant.from_documents(
        split_documents,
        embeddings,
        collection_name="financial_advisor_bm25_eval"
    )
    print("✅ Recreated vector store")
    
    print()
    return docs, dataset, vector_store, split_documents

def create_bm25_retriever(split_documents):
    """
    Create BM25 retriever
    """
    print("🔍 **Creating BM25 Retriever**")
    print("=" * 50)
    
    from langchain_community.retrievers import BM25Retriever
    
    # Create BM25 retriever from split documents
    bm25_retriever = BM25Retriever.from_documents(split_documents)
    bm25_retriever.k = 5  # Set number of documents to retrieve
    
    print("✅ BM25 retriever created")
    print()
    
    return bm25_retriever

def create_rag_chain(retriever):
    """
    Create RAG chain with the specified retriever
    """
    print("🔗 **Creating RAG Chain with BM25 Retriever**")
    print("=" * 60)
    
    # Create LLM for generation
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    # Create RAG prompt
    RAG_PROMPT = """\
You are a financial advisor assistant who answers questions based on provided context. 
You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}

Answer:"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    
    # Create RAG functions
    def retrieve(state):
        retrieved_docs = retriever.get_relevant_documents(state["question"])
        return {"context": retrieved_docs}

    def generate(state):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
        response = llm.invoke(messages)
        return {"response": response.content}
    
    # Build graph
    class State(TypedDict):
        question: str
        context: List[Document]
        response: str

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    print("✅ RAG Chain created successfully")
    print()
    
    return graph

def run_evaluation(dataset, graph):
    """
    Run evaluation on the dataset using the RAG chain
    """
    print("🔄 **Running Evaluation on Dataset**")
    print("=" * 50)
    
    print("🔄 Running evaluation on generated dataset...")
    
    for test_row in dataset:
        try:
            response = graph.invoke({"question": test_row.user_input})
            test_row.response = response["response"]
            test_row.retrieved_contexts = [context.page_content for context in response["context"]]
            print(f"  ✅ Processed: {test_row.user_input[:50]}...")
        except Exception as e:
            print(f"⚠️ Error processing: {e}")
            test_row.response = "Error generating response"
            test_row.retrieved_contexts = []
    
    print("✅ Evaluation complete")
    print()
    
    return dataset

def convert_to_evaluation_dataset(dataset):
    """
    Convert to Evaluation Dataset
    """
    print("🔄 **Converting to Evaluation Dataset**")
    print("=" * 50)
    
    evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())
    print("✅ Converted to EvaluationDataset")
    print()
    
    return evaluation_dataset

def run_ragas_evaluation(evaluation_dataset):
    """
    Run RAGAS Evaluation
    """
    print("📊 **Running RAGAS Evaluation**")
    print("=" * 50)
    
    # Set up evaluator
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
    
    print("📈 Running RAGAS evaluation...")
    
    try:
        custom_run_config = RunConfig(timeout=360)
        
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[Faithfulness(), ResponseRelevancy(), ContextPrecision(), ContextRecall()],
            llm=evaluator_llm,
            run_config=custom_run_config
        )
        
        print("✅ RAGAS evaluation complete!")
        return result
        
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_results(result):
    """
    Display Results
    """
    print("📊 **BM25 Retriever Evaluation Results**")
    print("=" * 60)
    
    if result is None:
        print("❌ No results to display")
        return
    
    try:
        # Simple output - just print the result directly
        print("Evaluation Results:")
        print(result)
    except Exception as e:
        print(f"❌ Error displaying results: {e}")
        print(f"Result type: {type(result)}")
        print("Raw result content:")
        print(str(result))
    
    return result

def save_results(dataset, result):
    """
    Save Results
    """
    print("💾 **Saving Results**")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Save evaluation results
    if result:
        try:
            # Try to convert result to a serializable format
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            elif hasattr(result, '__dict__'):
                result_dict = result.__dict__
            elif isinstance(result, dict):
                result_dict = result
            else:
                # Fallback: convert to string representation
                result_dict = {"raw_result": str(result)}
            
            results_filename = f"results/bm25_results_{timestamp}.json"
            with open(results_filename, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            print(f"💾 Evaluation results saved: {results_filename}")
            
        except Exception as e:
            print(f"⚠️ Error saving results: {e}")
            # Save as string as fallback
            results_filename = f"results/bm25_results_{timestamp}.txt"
            with open(results_filename, 'w') as f:
                f.write(str(result))
            print(f"💾 Evaluation results saved as text: {results_filename}")
    
    print("✅ All results saved!")
    print()

def main():
    """
    Main function - evaluate BM25 retriever
    """
    print("🚀 **BM25 Retriever Evaluation**")
    print("=" * 70)
    print("Evaluating BM25 retrieval with RAGAS metrics")
    print()
    
    try:
        # Load shared state
        docs, dataset, vector_store, split_documents = load_shared_state()
        if docs is None:
            return
        
        # Create BM25 retriever
        retriever = create_bm25_retriever(split_documents)
        
        # Create RAG chain
        graph = create_rag_chain(retriever)
        
        # Run evaluation
        dataset = run_evaluation(dataset, graph)
        
        # Convert to evaluation dataset
        evaluation_dataset = convert_to_evaluation_dataset(dataset)
        
        # Run RAGAS evaluation
        result = run_ragas_evaluation(evaluation_dataset)
        
        # Display results
        display_results(result)
        
        # Save results
        save_results(dataset, result)
        
        print("✅ **BM25 Retriever Evaluation Complete!**")
        print("📸 **Ready for screenshots!**")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 