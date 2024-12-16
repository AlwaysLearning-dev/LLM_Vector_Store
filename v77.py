"""
title: Qdrant Sigma Rules Pipeline
author: open-webui
date: 2024-12-14
version: 1.0
license: MIT
description: A pipeline for searching and analyzing Sigma rules using Qdrant and LLM
requirements: qdrant-client, transformers, requests, torch
"""

from typing import List, Dict, Any, Generator
import logging
import json
import re
import os
import requests
import torch

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_COLLECTION: str
        LLM_MODEL_NAME: str
        LLM_BASE_URL: str
        ENABLE_CONTEXT: bool
        LOG_LEVEL: str
        LLAMA_MODEL_PATH: str

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
                "LLAMA_MODEL_PATH": os.getenv("LLAMA_MODEL_PATH", "/home/bob/llama/"),
            }
        )

        logging.basicConfig(level=self.valves.LOG_LEVEL)
        self.logger = logging.getLogger("Pipeline")

        self.qdrant = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )

        # Load local LLaMA model (same path as used in ingestion script)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.valves.LLAMA_MODEL_PATH)
        self.model = AutoModel.from_pretrained(self.valves.LLAMA_MODEL_PATH, trust_remote_code=True).to(self.device)

    async def on_startup(self):
        """Recreate collection with matching vector size if desired."""
        # Optional: If you want to recreate the collection automatically here, do it:
        # test embedding dimension
        test_text = "test dimension check"
        test_embed = self.generate_embedding(test_text)
        embedding_size = len(test_embed)
        try:
            self.qdrant.recreate_collection(
                collection_name=self.valves.QDRANT_COLLECTION,
                vectors_config={
                    "default": {
                        "size": embedding_size,
                        "distance": "Cosine"
                    }
                }
            )
            collection_info = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
            print(f"Recreated Qdrant collection with {embedding_size}-dim vectors: {collection_info}")
        except Exception as e:
            print(f"Error recreating Qdrant collection: {e}")
            raise

    async def on_shutdown(self):
        pass

    def generate_embedding(self, text: str) -> list:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
            # Mean pooling
            embedding = outputs.mean(dim=1).squeeze()
        return embedding.cpu().tolist()

    def extract_search_terms(self, query: str) -> str:
        if query.startswith("search_qdrant:"):
            raw_query = query.replace("search_qdrant:", "", 1).strip()
            return raw_query if raw_query else query
        return query

    def vector_search_qdrant(self, search_text: str, top_k: int = 5) -> List[Dict]:
        if not search_text:
            return []
        try:
            embedding = self.generate_embedding(search_text)
            results = self.qdrant.search(
                collection_name=self.valves.QDRANT_COLLECTION,
                query_vector=embedding,
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
                    yaml_output.append("description: |")
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

        return "\n".join(yaml_output)

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
        return "\n".join(context)

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
