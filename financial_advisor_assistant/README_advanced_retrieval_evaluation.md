# Advanced Retrieval Evaluation System

This system provides a modular approach to evaluate different advanced retrieval methods using RAGAS metrics and LangSmith tracing.

## 🚀 Quick Start

### 1. **First time**: Run `prepare_shared_state.py` once
```bash
cd financial_advisor_assistant
source .venv/bin/activate
python prepare_shared_state.py
```

### 2. **Test LangSmith Tracing** (Optional but Recommended)
```bash
python test_langsmith.py
```
This will test if LangSmith tracing is working properly. You'll need a LangSmith API key.

### 3. **Run Individual Evaluations**
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

## 📁 File Structure

```
financial_advisor_assistant/
├── prepare_shared_state.py          # Prepares shared resources
├── evaluate_multi_query.py          # Multi-Query retriever evaluation
├── evaluate_ensemble.py             # Ensemble retriever evaluation  
├── evaluate_parent_document.py      # Parent Document retriever evaluation
├── evaluate_bm25.py                 # BM25 retriever evaluation
├── test_langsmith.py               # LangSmith tracing test
├── shared_state/                   # Shared resources (created by prepare_shared_state.py)
│   ├── documents.pkl
│   ├── split_documents.pkl
│   ├── golden_dataset.csv
│   └── vector_store_info.pkl
└── results/                        # Evaluation results (created by evaluation scripts)
    ├── multi_query_results_TIMESTAMP.json
    ├── ensemble_results_TIMESTAMP.json
    ├── parent_document_results_TIMESTAMP.json
    └── bm25_results_TIMESTAMP.json
```

## 🔧 LangSmith Setup

### Getting a LangSmith API Key
1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up or log in
3. Go to Settings → API Keys
4. Create a new API key

### Testing LangSmith
Before running evaluations, test LangSmith tracing:
```bash
python test_langsmith.py
```

### LangSmith Projects
Each evaluation script uses a different LangSmith project:
- `financial_advisor_shared_state_preparation`
- `financial_advisor_multi_query_evaluation`
- `financial_advisor_ensemble_evaluation`
- `financial_advisor_parent_document_evaluation`
- `financial_advisor_bm25_evaluation`

## 📊 Evaluation Metrics

Each evaluation uses these RAGAS metrics:
- **Faithfulness**: Measures if the response is faithful to the retrieved context
- **Response Relevancy**: Measures if the response is relevant to the question
- **Context Precision**: Measures if the retrieved context is precise
- **Context Recall**: Measures if the retrieved context covers the answer

## 💡 Benefits

### Efficiency
- **Shared Resources**: Documents, golden dataset, and vector store are created once
- **Isolated Failures**: If one retriever fails, others continue
- **No Redundant Work**: Expensive steps (document loading, dataset generation) run once

### LangSmith Integration
- **Separate Projects**: Each retriever has its own LangSmith project for clean separation
- **Detailed Traces**: Full tracing of LLM calls, retrievals, and evaluations
- **Cost Analysis**: Track costs and latency for each retriever
- **Debugging**: Easy to debug issues with detailed traces

### Modularity
- **Independent Scripts**: Each retriever evaluation is self-contained
- **Easy Extension**: Add new retrievers by copying and modifying existing scripts
- **Clear Separation**: Each script focuses on one retriever type

## 🔍 Troubleshooting

### LangSmith Issues
If LangSmith tracing isn't working:
1. Check your API key is correct
2. Run `python test_langsmith.py` to test
3. Check network connectivity
4. Verify LangSmith service is up

### Missing Dependencies
If you get import errors:
```bash
pip install langsmith langchain-openai ragas qdrant-client
```

### Shared State Issues
If shared state is missing:
```bash
python prepare_shared_state.py
```

## 📈 Expected Output

Each evaluation will produce results like:
```json
{
  "faithfulness": 0.9566,
  "response_relevancy": 0.8956, 
  "context_precision": 0.9583,
  "context_recall": 0.6247
}
```

Plus detailed traces in LangSmith for cost and latency analysis. 