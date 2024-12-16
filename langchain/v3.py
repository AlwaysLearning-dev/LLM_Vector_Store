"""
title: Qdrant Vector Store with LangChain Integration
description: A pipeline for managing and querying vector stores using Qdrant and LangChain
version: 2.0
requirements:
    - qdrant-client==1.7.0
    - langchain==0.1.0
    - langchain-community==0.0.16
    - pydantic==2.7.4
    - sentence-transformers==2.2.2
"""

from typing import List, Dict, Any, Generator, Optional
import os
import json
from pydantic import BaseModel, Field, ConfigDict
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class Pipeline:
    """Pipeline for vector store operations and LLM interactions."""
    
    class Settings(BaseModel):
        """Configuration settings using Pydantic v2 syntax."""
        model_config = ConfigDict(
            extra='allow',
            validate_assignment=True,
            arbitrary_types_allowed=True
        )
        
        qdrant_host: str = Field(default="qdrant")
        qdrant_port: int = Field(default=6333)
        qdrant_collection: str = Field(default="sigma_rules")
        llm_model_name: str = Field(default="llama3.2")
        ollama_base_url: str = Field(default="http://ollama:11434")
        enable_context: bool = Field(default=True)
        embedding_model: str = Field(default="all-MiniLM-L6-v2")
        chunk_size: int = Field(default=1000)
        chunk_overlap: int = Field(default=200)

    def __init__(self):
        """Initialize the pipeline components."""
        self.settings = self.Settings()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model
        )
        
        # Initialize Qdrant client and vector store
        self.qdrant_client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port
        )
        
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.settings.qdrant_collection,
            embeddings=self.embeddings
        )
        
        # Initialize LLM components
        self.llm = ChatOllama(
            model=self.settings.llm_model_name,
            base_url=self.settings.ollama_base_url
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )

    def format_rule(self, rule: Dict[str, Any]) -> str:
        """Format a Sigma rule as YAML string."""
        yaml_lines = []
        for key, value in rule.items():
            if isinstance(value, (dict, list)):
                yaml_lines.append(f"{key}:")
                formatted_value = json.dumps(value, indent=2)
                yaml_lines.extend(f"  {line}" for line in formatted_value.split('\n'))
            else:
                yaml_lines.append(f"{key}: {value}")
        return '\n'.join(yaml_lines)

    def search_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for documents in the vector store."""
        try:
            docs = self.vector_store.similarity_search(query, k=limit)
            results = []
            for doc in docs:
                if isinstance(doc.page_content, str):
                    try:
                        content = json.loads(doc.page_content)
                    except json.JSONDecodeError:
                        content = {"content": doc.page_content}
                else:
                    content = doc.page_content
                results.append(content)
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def process_chat(self, query: str) -> Dict[str, Any]:
        """Process a chat query using the LLM chain."""
        try:
            result = self.chain({"question": query})
            return {
                "answer": result["answer"],
                "sources": [doc.page_content for doc in result.get("source_documents", [])]
            }
        except Exception as e:
            return {"error": str(e)}

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Main pipeline processing function."""
        if not prompt:
            return
        
        try:
            # Handle search requests
            if prompt.lower().startswith(('search', 'find', 'show', 'list')):
                results = self.search_documents(prompt)
                yield f"Found {len(results)} relevant items:\n\n"
                for idx, result in enumerate(results, 1):
                    yield f"Result {idx}: {result.get('title', 'Untitled')}\n"
                    yield "```yaml\n"
                    yield self.format_rule(result)
                    yield "\n```\n\n"
                return

            # Handle chat/questions
            chat_result = self.process_chat(prompt)
            if "error" in chat_result:
                yield f"Error: {chat_result['error']}"
            else:
                yield chat_result["answer"]

        except Exception as e:
            yield f"Error: {str(e)}"

    async def on_startup(self):
        """Initialize connections on startup."""
        try:
            collection_info = self.qdrant_client.get_collection(
                self.settings.qdrant_collection
            )
            print(f"Connected to Qdrant collection: {collection_info}")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise

    async def on_shutdown(self):
        """Clean up resources on shutdown."""
        pass
