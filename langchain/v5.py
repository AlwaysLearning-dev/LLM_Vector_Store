#!/usr/bin/env python3
"""
Minimal pipeline script integrating:
 - LangChain for Qdrant vector retrieval
 - Ollama for LLM generation (used by OpenWebUI)
 - Environment variables for Qdrant and LLM endpoints
"""

import os
import json
import requests
from typing import List, Dict, Any, Generator
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

def search_qdrant(query: str,
                  qdrant_host: str,
                  qdrant_port: int,
                  collection_name: str) -> List[str]:
    """Simple LangChain-based Qdrant similarity search returning doc contents."""
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    docs = vector_store.similarity_search(query, k=5)
    return [doc.page_content for doc in docs]

def create_llm_prompt(query: str, context_docs: List[str]) -> str:
    """Basic prompt construction, injecting context from Qdrant docs."""
    context_section = "\n\n".join(f"Doc {i+1}:\n{text}" for i, text in enumerate(context_docs))
    prompt = (
        f"Context from Qdrant:\n{context_section}\n\n"
        f"User Query:\n{query}\n\n"
        "Please answer based on the above context. Be specific."
    )
    return prompt

def pipe(query: str = "") -> Generator[str, None, None]:
    """Generator for OpenWebUI to stream responses from Ollama."""
    if not query:
        return

    # Load env vars
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "sigma_rules")
    llm_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm_model_name = os.getenv("LLAMA_MODEL_NAME", "llama2")

    # Get context from Qdrant
    context_docs = search_qdrant(query, qdrant_host, qdrant_port, collection_name)
    llm_prompt = create_llm_prompt(query, context_docs) if context_docs else query

    # Stream LLM response
    try:
        resp = requests.post(
            f"{llm_base_url}/api/generate",
            json={"model": llm_model_name, "prompt": llm_prompt},
            stream=True
        )
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    yield data.get("response", "")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield f"Error: {str(e)}"

def run(prompt: str, **kwargs) -> List[Dict[str, Any]]:
    """OpenWebUI calls this 'run' function to get pipeline output."""
    results = list(pipe(query=prompt))
    if not results:
        return []
    return [{"text": "".join(results)}]

if __name__ == "__main__":
    # Simple local test. In production, OpenWebUI calls run().
    test_query = "process injection"
    for chunk in pipe(test_query):
        print(chunk, end="")
