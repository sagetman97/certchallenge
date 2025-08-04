
# 🚀 Financial Advisor Assistant - Local Setup & Usage

## Quick Start Guide

### Prerequisites
- Python 3.12+
- OpenAI API Key
- Tavily API Key (optional, for external search)
- Cohere API Key (optional, for reranking)
- LangSmith API Key (optional, for tracing)

### Installation & Setup

1. **Navigate to the project directory**:
```bash
cd financial_advisor_assistant
```

2. **Install dependencies using uv**:
```bash
uv sync
```

3. **Set up environment variables**:
```bash
cp env_example.txt .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# TAVILY_API_KEY=your_tavily_key (optional)
# COHERE_API_KEY=your_cohere_key (optional)
# LANGSMITH_API_KEY=your_langsmith_key (optional)
```

4. **Load the RAG documents**:
```bash
python load_data.py
```

5. **Run the application**:
```bash
chainlit run app.py
```

The application will be available at `http://localhost:8000`

## 🧪 Advanced Retrieval Evaluation with RAGAS

This project includes a comprehensive evaluation framework for testing different retrieval methods using RAGAS metrics.

### Setup for Evaluation

1. **Prepare shared resources** (run once):
```bash
cd financial_advisor_assistant
python prepare_shared_state.py
```

2. **Test LangSmith tracing** (optional but recommended):
```bash
python test_langsmith.py
```

### Running Individual Retrieval Evaluations

Each evaluation script tests a different advanced retrieval method:

```bash
# Multi-Query Retrieval
python evaluate_multi_query.py

# Ensemble Retrieval  
python evaluate_ensemble.py

# Parent Document Retrieval
python evaluate_parent_document.py

# BM25 Retrieval
python evaluate_bm25.py
```

### Understanding the Evaluation Files

- **`prepare_shared_state.py`**: Creates shared resources (documents, golden dataset, vector store) used by all evaluation scripts
- **`evaluate_*.py`**: Individual evaluation scripts for different retrieval methods
- **`ragas_evaluation_step_by_step.py`**: Detailed step-by-step RAGAS evaluation process
- **`run_golden_dataset_evaluation.py`**: Evaluation using a golden dataset approach
- **`evaluate_system.py`**: Comprehensive system evaluation

### Evaluation Metrics

Each evaluation provides these RAGAS metrics:
- **Faithfulness**: Measures if the response is faithful to the retrieved context
- **Response Relevancy**: Measures if the response is relevant to the question  
- **Context Precision**: Measures if the retrieved context is precise
- **Context Recall**: Measures if the retrieved context covers the answer

### Results

Evaluation results are saved in the `results/` directory with timestamps:
- `multi_query_results_TIMESTAMP.json`
- `ensemble_results_TIMESTAMP.json`
- `parent_document_results_TIMESTAMP.json`
- `bm25_results_TIMESTAMP.json`

### LangSmith Integration

Each evaluation uses separate LangSmith projects for clean separation:
- `financial_advisor_multi_query_evaluation`
- `financial_advisor_ensemble_evaluation`
- `financial_advisor_parent_document_evaluation`
- `financial_advisor_bm25_evaluation`

This provides detailed traces for cost analysis, latency tracking, and debugging.


