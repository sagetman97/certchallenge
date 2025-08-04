"""
Chainlit app for Financial Advisor LangGraph system.
Based on AIE7 course patterns for Chainlit integration with LangGraph.
"""

import chainlit as cl
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from config.settings import settings
from agents.graph import process_query_with_langgraph
from agents.teams.tools import initialize_vector_store
import os


# Initialize Qdrant client
qdrant_client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT
)

# Initialize vector store
initialize_vector_store(qdrant_client)

# Initialize LLM
llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    temperature=settings.TEMPERATURE,
    max_tokens=settings.MAX_TOKENS
)


@cl.on_chat_start
async def start():
    """
    Initialize the chat session.
    Based on AIE7 course patterns for Chainlit chat initialization.
    """
    await cl.Message(
        content="🧠 **Financial Advisor Life Insurance Assistant**\n\n"
        "I'm your AI assistant powered by LangGraph with specialized capabilities:\n\n"
        "🔍 **Knowledge Base**: Answer questions about life insurance products and best practices\n"
        "🌐 **Current Research**: Get up-to-date market information and rates\n"
        "📊 **Needs Calculator**: Calculate coverage needs using multiple methodologies\n\n"
        "What I can do:\n"
        "• Answer questions about life insurance products and best practices\n"
        "• Calculate coverage needs using multiple methodologies\n"
        "• Research current market conditions and rates\n"
        "• Provide confidence-based responses with external search when needed\n\n"
        "How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Process user messages through the LangGraph system.
    Based on AIE7 course patterns for message processing.
    """
    try:
        # Get session context
        session_context = cl.user_session.get("session_context", {})
        uploaded_files = cl.user_session.get("uploaded_files", [])
        
        print(f"🔍 **MESSAGE PROCESSING**")
        print(f"📝 Query: {message.content}")
        print(f"📁 Uploaded Files: {len(uploaded_files)} files")
        for file in uploaded_files:
            print(f"   - {file.get('name', 'Unknown')}: {file.get('path', 'No path')}")
        
        # Check if user has uploaded files and is asking about portfolio analysis
        query_lower = message.content.lower()
        has_uploaded_files = len(uploaded_files) > 0
        is_portfolio_query = any(word in query_lower for word in ["portfolio", "analyze", "upload", "csv", "diversify"])
        
        print(f"📊 **CONTEXT ANALYSIS**")
        print(f"   Has Uploaded Files: {has_uploaded_files}")
        print(f"   Is Portfolio Query: {is_portfolio_query}")
        
        if has_uploaded_files and is_portfolio_query:
            print(f"✅ **PORTFOLIO CONTEXT DETECTED**")
            # Add portfolio context to session
            session_context["has_uploaded_files"] = True
            session_context["uploaded_files"] = uploaded_files
            cl.user_session.set("session_context", session_context)
        
        # Process query through LangGraph
        result = process_query_with_langgraph(
            llm=llm,
            query=message.content,
            session_context=session_context
        )
        
        # Extract response and metadata
        response = result.get("response", "I'm unable to provide a response at this time.")
        metadata = result.get("metadata", {})
        
        # Update session context
        if metadata.get("session_context"):
            cl.user_session.set("session_context", metadata["session_context"])
        
        # Send response
        await cl.Message(
            content=response,
            author="Assistant"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error Processing Request**\n\nI encountered an error: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists.",
            author="Assistant"
        ).send()


@cl.on_chat_end
async def end():
    """
    Clean up when chat session ends.
    Based on AIE7 course patterns for session cleanup.
    """
    # Clear session context
    cl.user_session.set("session_context", {})
    print("Cleared session documents for session")


# Note: File upload functionality will be added in a future update
# when the appropriate Chainlit decorator is available 