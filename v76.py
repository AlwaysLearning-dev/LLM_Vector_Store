"""
title: Qdrant Sigma Rules Pipeline
author: open-webui
date: 2024-12-14
version: 1.0
license: MIT
description: A pipeline for searching and analyzing Sigma rules using Qdrant and LLM
requirements: qdrant-client, sentence-transformers, requests
"""

from typing import List, Dict, Any, Generator
import logging
import json
import re
import os
import requests

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

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

    def __init__(self):
        self.valves = self.Valves(
            **{
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sigma_rules"),
                "LLM_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                "LLM_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                "ENABLE_CONTEXT": os.getenv("ENABLE_CONTEXT", "true").lower() == "true",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            }
        )

        logging.basicConfig(level=self.valves.LOG_LEVEL)
        self.logger = logging.getLogger("Pipeline")

        self.qdrant = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )
        self.embedding_model = SentenceTransformer(self.valves.EMBEDDING_MODEL)

    async def on_startup(self):
        """Recreate collection with specified vector size and distance if needed."""
        try:
            # Recreate the collection with size=3072, COSINE distance
            self.qdrant.recreate_collection(
                collection_name=self.valves.QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=3072,  # Must match your embedding model dimension
                    distance=qmodels.Distance.COSINE
                )
            )
            collection_info = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
            print(f"Recreated Qdrant collection: {collection_info}")
        except Exception as e:
            print(f"Error recreating Qdrant collection: {e}")
            raise

    async def on_shutdown(self):
        """Clean up resources."""
        pass

    def extract_search_terms(self, query: str) -> str:
        if query.startswith("search_qdrant:"):
            raw_query = query.replace("search_qdrant:", "", 1).strip()
            return raw_query if raw_query else query
        return query

    def vector_search_qdrant(self, search_text: str, top_k: int = 5) -> List[Dict]:
        if not search_text:
            return []
        try:
            embeddings = self.embedding_model.encode([search_text])[0]
            results = self.qdrant.search(
                collection_name=self.valves.QDRANT_COLLECTION,
                query_vector=embeddings.tolist(),
                limit=top_k,
                with_payload=True
            )
            matches = []
            for r in results:
                if r and r.payload:
                    matches.append(r.payload)
            return matches
        except Exception as e:
            self.logger.error(f"Qdrant vector search error: {e}")
            return []

    def format_rule(self, rule: Dict) -> str:
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
                yaml_output.append("")
            elif field in ['logsource', 'detection']:
                yaml_output.append(f"{field}:")
                if isinstance(value, dict):
                    dict_lines = json.dumps(value, indent=4).split('\n')
                    for line in dict_lines[1:-1]:
                        yaml_output.append(f"    {line.strip()}")
                else:
                    yaml_output.append("    {}")
                yaml_output.append("")
            elif isinstance(value, list):
                yaml_output.append(f"{field}:")
                if value:
                    for item in value:
                        yaml_output.append(f"    - {item}")
                else:
                    yaml_output.append("    - none")
                yaml_output.append("")
            elif isinstance(value, dict):
                yaml_output.append(f"{field}:")
                if value:
                    dict_lines = json.dumps(value, indent=4).split('\n')
                    for line in dict_lines[1:-1]:
                        yaml_output.append(f"    {line.strip()}")
                else:
                    yaml_output.append("    {}")
                yaml_output.append("")
            else:
                yaml_output.append(f"{field}: {value if value is not None else 'none'}")

        return '\n'.join(yaml_output)

    def get_context_from_rules(self, rules: List[Dict]) -> str:
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

    def create_llm_prompt(self, query: str, context: str) -> str:
        return f"""You have access to Sigma rule data. Here are some relevant Sigma rules for context:

{context}

Based on these rules, please address the following:

{query}

When applicable, reference the relevant rule details.
"""

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            search_text = self.extract_search_terms(query)
            matches = self.vector_search_qdrant(search_text, top_k=5)

            context = ""
            if matches and self.valves.ENABLE_CONTEXT:
                context = self.get_context_from_rules(matches)

            llm_prompt = self.create_llm_prompt(query, context)

            response = requests.post(
                url=f"{self.valves.LLM_BASE_URL}/api/generate",
                json={"model": self.valves.LLM_MODEL_NAME, "prompt": llm_prompt},
                stream=True
            )
            if not response.ok:
                yield f"LLM request failed: {response.text}"
                return

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
        results = list(self.pipe(prompt=prompt, **kwargs))
        if not results:
            return []
        return [{"text": "".join(results)}]
