"""
Step-by-Step RAGAS Evaluation for Financial Advisor Assistant

This script breaks down the RAGAS evaluation process into clear steps
that can be run independently. Each section can be uncommented and run
as needed.
"""

import sys
import os
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
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
import pandas as pd
import json
from datetime import datetime

# RAGAS imports
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall

# Import settings
from config.settings import settings

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
            docs[:10],  # Use first 10 documents for generation (reduced from 20)
            testset_size=5  # Generate 5 test cases (reduced from 10)
        )
    except Exception as e:
        print(f"⚠️ RAGAS generation failed: {e}")
        print("🔄 Trying with even smaller dataset...")
        try:
            dataset = generator.generate_with_langchain_docs(
                docs[:5],  # Use first 5 documents for generation
                testset_size=3  # Generate 3 test cases
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

def step_4_create_rag_chain(docs):
    """
    Step 4: Create RAG Chain
    """
    print("🔗 **Step 4: Creating RAG Chain**")
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
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create LLM for generation (using normal GPT-4 as in our application)
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
    
    return graph, retriever, rag_prompt, llm, vector_store

def step_5_test_rag_chain(graph):
    """
    Step 5: Test RAG Chain
    """
    print("🧪 **Step 5: Testing RAG Chain**")
    print("=" * 50)
    
    test_question = "What is the difference between term and whole life insurance?"
    print(f"Testing with question: {test_question}")
    
    response = graph.invoke({"question": test_question})
    print("\nResponse:")
    print(response["response"])
    print()
    
    return response

def step_6_run_evaluation(dataset, graph):
    """
    Step 6: Run Evaluation on Dataset
    """
    print("📊 **Step 6: Running Evaluation on Dataset**")
    print("=" * 60)
    
    print("🔄 Running evaluation on generated dataset...")
    
    for test_row in dataset:
        try:
            response = graph.invoke({"question": test_row.eval_sample.user_input})
            test_row.eval_sample.response = response["response"]
            test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]
            print(f"  ✅ Processed: {test_row.eval_sample.user_input[:50]}...")
        except Exception as e:
            print(f"⚠️ Error processing: {e}")
            test_row.eval_sample.response = "Error generating response"
            test_row.eval_sample.retrieved_contexts = []
    
    print("✅ Evaluation complete")
    print()
    
    return dataset

def step_7_convert_to_evaluation_dataset(dataset):
    """
    Step 7: Convert to Evaluation Dataset
    """
    print("🔄 **Step 7: Converting to Evaluation Dataset**")
    print("=" * 50)
    
    evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())
    print("✅ Converted to EvaluationDataset")
    print()
    
    return evaluation_dataset

def step_8_run_ragas_evaluation(evaluation_dataset):
    """
    Step 8: Run RAGAS Evaluation
    """
    print("📊 **Step 8: Running RAGAS Evaluation**")
    print("=" * 50)
    
    # Set up evaluator (following original notebook exactly)
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

def step_9_display_results(result):
    """
    Step 9: Display Results
    """
    print("📊 **Step 9: RAGAS Evaluation Results**")
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

def step_10_save_results(dataset, result):
    """
    Step 10: Save Results
    """
    print("💾 **Step 10: Saving Results**")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save golden dataset
    dataset_filename = f"golden_dataset_{timestamp}.csv"
    dataset.to_pandas().to_csv(dataset_filename, index=False)
    print(f"💾 Golden dataset saved: {dataset_filename}")
    
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
            
            results_filename = f"evaluation_results_{timestamp}.json"
            with open(results_filename, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            print(f"💾 Evaluation results saved: {results_filename}")
            
        except Exception as e:
            print(f"⚠️ Error saving results: {e}")
            # Save as string as fallback
            results_filename = f"evaluation_results_{timestamp}.txt"
            with open(results_filename, 'w') as f:
                f.write(str(result))
            print(f"💾 Evaluation results saved as text: {results_filename}")
    
    print("✅ All results saved!")
    print()

def step_11_advanced_retrieval_testing(dataset, vector_store):
    """
    Step 11: Advanced Retrieval Testing (placeholder for future implementation)
    """
    print("🚀 **Step 11: Advanced Retrieval Testing**")
    print("=" * 60)
    
    print("📊 This step will be implemented later with Cohere rerank")
    print("   For now, we're focusing on baseline RAG evaluation")
    print()
    
    return None

def main():
    """
    Main function - run all steps
    """
    print("🚀 **RAGAS Evaluation for Financial Advisor Assistant**")
    print("=" * 70)
    print("Running all steps sequentially...")
    print()
    
    try:
        # Step 1: Setup
        step_1_setup_environment()
        
        # Step 2: Load documents
        docs = step_2_load_documents()
        
        # Step 3: Create golden dataset
        dataset = step_3_create_golden_dataset(docs)
        
        # Step 4: Create RAG chain
        graph, retriever, rag_prompt, llm, vector_store = step_4_create_rag_chain(docs)
        
        # Step 5: Test RAG chain
        step_5_test_rag_chain(graph)
        
        # Step 6: Run evaluation
        dataset = step_6_run_evaluation(dataset, graph)
        
        # Step 7: Convert to evaluation dataset
        evaluation_dataset = step_7_convert_to_evaluation_dataset(dataset)
        
        # Step 8: Run RAGAS evaluation
        result = step_8_run_ragas_evaluation(evaluation_dataset)
        
        # Step 9: Display results
        step_9_display_results(result)
        
        # Step 10: Save results
        step_10_save_results(dataset, result)
        
        # Step 11: Advanced retrieval testing (skipped for now - will be added later)
        print("⏭️ **Step 11: Advanced retrieval testing skipped**")
        print("   Will be implemented later with Cohere rerank")
        print()
        
        print("✅ **RAGAS Evaluation Complete!**")
        print("📸 **Ready for screenshots!**")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Uncomment the step you want to run, or run main() for everything
    
    # Individual steps (uncomment to run specific steps):
    # step_1_setup_environment()
    # docs = step_2_load_documents()
    # dataset = step_3_create_golden_dataset(docs)
    # graph, retriever, rag_prompt, llm, vector_store = step_4_create_rag_chain(docs)
    # step_5_test_rag_chain(graph)
    # dataset = step_6_run_evaluation(dataset, graph)
    # evaluation_dataset = step_7_convert_to_evaluation_dataset(dataset)
    # result = step_8_run_ragas_evaluation(evaluation_dataset)
    # step_9_display_results(result)
    # step_10_save_results(dataset, result)
    # step_11_advanced_retrieval_testing(dataset, vector_store)  # Placeholder for future
    
    # Or run everything:
    main() 