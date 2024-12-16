"""
title: Simplified Qdrant-LangChain Pipeline for Sigma Rules
author: open-webui
date: 2024-12-16
version: 2.0
license: MIT
description: A simplified pipeline using LangChain for context handling with Sigma rules
requirements: qdrant-client, langchain, pydantic
"""

from typing import List, Dict, Any, Generator
import os
import json
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class Pipeline:
    class Config(BaseModel):
        qdrant_host: str = "qdrant"
        qdrant_port: int = 6333
        qdrant_collection: str = "sigma_rules"
        llm_model_name: str = "llama3.2"
        ollama_base_url: str = "http://ollama:11434"
        enable_context: bool = True

    def __init__(self):
        self.config = self.Config()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port
        )
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize vector store
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.config.qdrant_collection,
            embeddings=self.embeddings
        )
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=self.config.llm_model_name,
            base_url=self.config.ollama_base_url
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )

    def format_rule(self, rule: Dict) -> str:
        """Format a single Sigma rule as YAML."""
        yaml_lines = []
        for key, value in rule.items():
            if isinstance(value, (dict, list)):
                yaml_lines.append(f"{key}:")
                formatted_value = json.dumps(value, indent=2)
                yaml_lines.extend(f"  {line}" for line in formatted_value.split('\n'))
            else:
                yaml_lines.append(f"{key}: {value}")
        return '\n'.join(yaml_lines)

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and generate responses."""
        if not prompt:
            return
        
        try:
            # Handle direct search requests
            if prompt.lower().startswith(('search', 'find', 'show', 'list')):
                docs = self.vector_store.similarity_search(prompt)
                yield f"Found {len(docs)} relevant Sigma rules:\n\n"
                for i, doc in enumerate(docs, 1):
                    rule = doc.page_content
                    if isinstance(rule, str):
                        rule = json.loads(rule)
                    yield f"Rule {i}: {rule.get('title', 'Untitled')}\n"
                    yield "```yaml\n"
                    yield self.format_rule(rule)
                    yield "\n```\n\n"
                return

            # Handle questions using LangChain conversation chain
            result = self.chain({"question": prompt})
            yield result["answer"]

        except Exception as e:
            yield f"Error: {str(e)}"

    async def on_startup(self):
        """Verify connections on startup."""
        try:
            collection_info = self.qdrant_client.get_collection(
                self.config.qdrant_collection
            )
            print(f"Connected to Qdrant collection: {collection_info}")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise

    async def on_shutdown(self):
        """Clean up resources."""
        pass
