"""
title: Qdrant Sigma Rules Pipeline
author: open-webui
date: 2024-12-14
version: 1.2
license: MIT
description: A pipeline for searching and analyzing Sigma rules using Qdrant and LLM with vector semantic search
requirements: qdrant-client, requests, sentence-transformers
"""

from typing import List, Dict, Any, Generator
import logging
import json
import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import UpdateResult
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_COLLECTION: str
        LLM_MODEL_NAME: str
        LLM_BASE_URL: str
        ENABLE_CONTEXT: bool
        LOG_LEVEL: str
        EMBEDDING_MODEL: str
        VECTOR_SIZE: int
        SEMANTIC_SEARCH_LIMIT: int

    def __init__(self):
        # Initialize valves with environment variables or defaults
        self.valves = self.Valves(
            **{
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sigma_rules"),
                "LLM_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                "LLM_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                "ENABLE_CONTEXT": os.getenv("ENABLE_CONTEXT", "true").lower() == "true",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                "VECTOR_SIZE": int(os.getenv("VECTOR_SIZE", "384")),
                "SEMANTIC_SEARCH_LIMIT": int(os.getenv("SEMANTIC_SEARCH_LIMIT", "20"))
            }
        )

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(self.valves.EMBEDDING_MODEL)

    async def on_startup(self):
        """Verify connections and create collection if needed."""
        try:
            # Check if collection exists
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            # If collection doesn't exist, create it with vector configuration
            if self.valves.QDRANT_COLLECTION not in collection_names:
                self.qdrant.recreate_collection(
                    collection_name=self.valves.QDRANT_COLLECTION,
                    vectors_config={
                        "rule_vector": VectorParams(
                            size=self.valves.VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    }
                )
                print(f"Created new collection: {self.valves.QDRANT_COLLECTION}")
            
            collection_info = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
            print(f"Connected to Qdrant collection: {collection_info}")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise

    async def on_shutdown(self):
        """Clean up resources."""
        pass

    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text using sentence-transformers."""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def search_qdrant(self, query: str) -> List[Dict]:
        """Search for rules using semantic search."""
        try:
            query_vector = self.get_text_embedding(query)
            if not query_vector:
                return []

            # Search with named vectors
            results = self.qdrant.search(
                collection_name=self.valves.QDRANT_COLLECTION,
                query_vector=("rule_vector", query_vector),
                limit=self.valves.SEMANTIC_SEARCH_LIMIT,
                with_payload=True
            )
            
            return [hit.payload for hit in results]

        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    def upsert_rule(self, rule: Dict) -> UpdateResult:
        """Insert or update a rule with vector embedding."""
        try:
            # Create searchable text from rule fields
            searchable_text = f"{rule.get('title', '')} {rule.get('description', '')} "
            if rule.get('detection'):
                searchable_text += json.dumps(rule['detection'])
            
            # Generate vector embedding
            vector = self.get_text_embedding(searchable_text)
            if not vector:
                raise ValueError("Failed to generate embedding")

            # Create point with named vectors
            point = PointStruct(
                id=hash(rule.get('id', '') or rule.get('title', '')),
                vector={"rule_vector": vector},
                payload=rule
            )

            # Upsert the point
            operation_info = self.qdrant.upsert(
                collection_name=self.valves.QDRANT_COLLECTION,
                points=[point]
            )
            
            return operation_info

        except Exception as e:
            print(f"Error upserting rule: {e}")
            raise

    def format_rule(self, rule: Dict) -> str:
        """Format rule in Sigma YAML."""
        yaml_output = []
        fields = [
            'title', 'id', 'status', 'description', 'references', 'author',
            'date', 'modified', 'tags', 'logsource', 'detection',
            'falsepositives', 'level', 'filename'
        ]
        
        for field in fields:
            value = rule.get(field)
            if field == 'description':
                if value:
                    yaml_output.append(f"description: |")
                    for line in str(value).split('\n'):
                        yaml_output.append(f"    {line.strip()}")
                else:
                    yaml_output.append("description: |")
                    yaml_output.append("    No description provided")
                    
            elif field in ['logsource', 'detection']:
                yaml_output.append(f"{field}:")
                if isinstance(value, dict):
                    dict_lines = json.dumps(value, indent=4).split('\n')
                    for line in dict_lines[1:-1]:
                        yaml_output.append(f"    {line.strip()}")
                else:
                    yaml_output.append("    {}")
                    
            elif isinstance(value, list):
                yaml_output.append(f"{field}:")
                if value:
                    for item in value:
                        yaml_output.append(f"    - {item}")
                else:
                    yaml_output.append("    - none")
                    
            elif isinstance(value, dict):
                yaml_output.append(f"{field}:")
                if value:
                    dict_lines = json.dumps(value, indent=4).split('\n')
                    for line in dict_lines[1:-1]:
                        yaml_output.append(f"    {line.strip()}")
                else:
                    yaml_output.append("    {}")
                    
            else:
                yaml_output.append(f"{field}: {value if value is not None else 'none'}")
                
            if field in ['description', 'detection', 'logsource']:
                yaml_output.append("")
                
        return '\n'.join(yaml_output)

    def get_context_from_rules(self, rules: List[Dict]) -> str:
        """Create context string from rules for LLM."""
        context = []
        for idx, rule in enumerate(rules, 1):
            context.append(f"Rule {idx}:")
            context.append(f"Title: {rule.get('title', 'Untitled')}")
            if rule.get('description'):
                context.append(f"Description: {rule['description']}")
            if rule.get('detection'):
                context.append("Detection:")
                context.append(json.dumps(rule['detection'], indent=2))
            context.append("---")
        return '\n'.join(context)

    def looks_like_search(self, query: str) -> bool:
        """Check if query is a search request."""
        search_words = ['search', 'find', 'show', 'list', 'get']
        query_words = query.lower().split()
        return any(word in query_words for word in search_words) or len(query_words) <= 3

    def looks_like_question(self, query: str) -> bool:
        """Check if query is a question about rules."""
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain']
        contains_about = 'about' in query.lower()
        starts_with_question = any(query.lower().startswith(word) for word in question_words)
        return starts_with_question or contains_about

    def create_llm_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM that includes context."""
        return f"""Here are some relevant Sigma detection rules for context:

{context}

Based on these rules, please answer this question:
{query}

Please be specific and refer to the rules when applicable."""

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and return results."""
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            # Perform semantic search
            matches = self.search_qdrant(query)
            
            # If it's a direct search request, show the rules
            if self.looks_like_search(query):
                if matches:
                    yield f"Found {len(matches)} semantically relevant Sigma rules:\n\n"
                    for idx, rule in enumerate(matches, 1):
                        # Rule number and title outside of code block
                        yield f"Rule {idx}: {rule.get('title', 'Untitled')}\n"
                        # Rule content in code block
                        yield "```yaml\n"
                        yield self.format_rule(rule)
                        yield "\n```\n\n"
                    return
                else:
                    yield "No semantically similar Sigma rules found.\n"
                    return
            
            # If it's a question and context is enabled, include matching rules
            if self.looks_like_question(query) and matches and self.valves.ENABLE_CONTEXT:
                context = self.get_context_from_rules(matches)
                llm_prompt = self.create_llm_prompt(query, context)
            else:
                llm_prompt = query

            # Get LLM response
            response = requests.post(
                url=f"{self.valves.LLM_BASE_URL}/api/generate",
                json={"model": self.valves.LLM_MODEL_NAME, "prompt": llm_prompt},
                stream=True
            )
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get("response", "")
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            yield f"Error: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run pipeline and return results."""
        results = list(self.pipe(prompt=prompt, **kwargs))
        if not results:
            return []
        return [{"text": "".join(results)}]
