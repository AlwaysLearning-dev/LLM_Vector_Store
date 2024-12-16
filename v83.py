"""
title: Qdrant + LangChain Sigma Rules Pipeline
author: open-webui
date: 2024-12-15
version: 1.0
license: MIT
description: Simplified pipeline for searching and analyzing Sigma rules using Qdrant, LangChain, and an LLM with conversation context
requirements:
  - langchain
  - qdrant-client
  - requests
  - pydantic
"""

import os
import json
import logging
import re
import requests
from typing import List, Dict, Any, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from pydantic import BaseModel
from qdrant_client import QdrantClient
# LangChain imports
from langchain.vectorstores import Qdrant as QdrantStore
from langchain.embeddings import HuggingFaceEmbeddings

@dataclass
class ConversationContext:
    last_rules: List[Dict] = field(default_factory=list)
    last_query: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def is_valid(self, timeout_minutes: int = 5) -> bool:
        """Check if context is still valid within timeout window."""
        return datetime.now() - self.timestamp < timedelta(minutes=timeout_minutes)

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_COLLECTION: str
        LLM_MODEL_NAME: str
        LLM_BASE_URL: str
        ENABLE_CONTEXT: bool
        CONTEXT_TIMEOUT: int
        LOG_LEVEL: str
        SEARCH_PREFIX: str

    def __init__(self):
        self.valves = self.Valves(
            **{
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sigma_rules"),
                "LLM_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                "LLM_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                "ENABLE_CONTEXT": True,
                "CONTEXT_TIMEOUT": int(os.getenv("CONTEXT_TIMEOUT", "5")),
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "SEARCH_PREFIX": os.getenv("SEARCH_PREFIX", "search_qdrant:")
            }
        )

        logging.basicConfig(level=getattr(logging, self.valves.LOG_LEVEL))
        self.logger = logging.getLogger(__name__)

        self.context = ConversationContext()

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )

        # LangChain: embedding + vector store
        # Here we assume embeddings match what you used for ingestion (e.g. 'all-MiniLM-L6-v2')
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = QdrantStore(
            client=self.qdrant_client,
            collection_name=self.valves.QDRANT_COLLECTION,
            embeddings=self.embeddings,
            # If your points were stored under default vector name, you don't need vector_name here
        )

    def extract_search_terms(self, query: str) -> str:
        """
        Convert user query into a single search string.
        If it starts with SEARCH_PREFIX, strip that off.
        """
        search_prefix = self.valves.SEARCH_PREFIX
        if query.startswith(search_prefix):
            return query[len(search_prefix):].strip()
        # If quoted phrases, unify them; else just return the raw query
        phrases = re.findall(r'"([^"]*)"', query)
        if phrases:
            return " ".join(phrases)
        return query

    def get_langchain_docs(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Use LangChain QdrantStore to retrieve relevant docs (Sigma rules).
        Each doc.page_content is the original text. We also stored metadata as doc.metadata.
        """
        docs = self.vectorstore.similarity_search(query, k=top_k)
        # Convert LangChain docs to a list of dict payloads
        results = []
        for doc in docs:
            # doc.metadata should hold original Sigma fields
            meta = doc.metadata if doc.metadata else {}
            results.append(meta)
        return results

    def get_referenced_rules(self, query: str) -> List[Dict]:
        """If user references rules from context, retrieve them."""
        if not self.context.is_valid(self.valves.CONTEXT_TIMEOUT):
            return []
        # Simple reference check
        match = re.search(r'rule\s+(\d+)', query.lower())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(self.context.last_rules):
                return [self.context.last_rules[idx]]
        return []

    def get_llm_response(self, prompt: str) -> Generator[str, None, None]:
        """Stream response from the LLM API."""
        try:
            response = requests.post(
                url=f"{self.valves.LLM_BASE_URL}/api/generate",
                json={"model": self.valves.LLM_MODEL_NAME, "prompt": prompt},
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
            self.logger.error(f"LLM request error: {e}")
            yield f"Error: {str(e)}"

    def format_rule(self, rule: Dict) -> str:
        """Minimal Sigma rule formatting."""
        out = []
        fields = ["title","description","author","date","tags","logsource","detection"]
        for f in fields:
            val = rule.get(f, None)
            if isinstance(val, dict) or isinstance(val, list):
                val = json.dumps(val, indent=2)
            out.append(f"{f}: {val}")
        return "\n".join(out)

    def create_llm_prompt(self, query: str, rules: List[Dict]) -> str:
        if not rules:
            return query
        context_lines = []
        for i, rule in enumerate(rules, 1):
            context_lines.append(f"Rule {i}: {rule.get('title','Untitled')}")
            if rule.get('description'):
                context_lines.append(f"Description: {rule['description']}")
        context = "\n".join(context_lines)
        return f"""You have access to these Sigma rules:

{context}

Answer the user's question based on these rules:
{query}
"""

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        query = prompt or kwargs.get('user_message', '')
        if not query.strip():
            return

        try:
            # If the query references a previously shown rule
            ref_rules = self.get_referenced_rules(query)
            if ref_rules:
                llm_prompt = self.create_llm_prompt(query, ref_rules)
                yield from self.get_llm_response(llm_prompt)
                return

            # Otherwise, retrieve from Qdrant via LangChain
            search_text = self.extract_search_terms(query)
            docs = self.get_langchain_docs(search_text, top_k=5)

            if docs:
                self.context.last_rules = docs
                self.context.last_query = query
                self.context.timestamp = datetime.now()

            # If user explicitly searching, just show them
            search_cmds = ["search", "find", "show", "list"]
            if any(cmd in query.lower().split() for cmd in search_cmds) or len(query.split()) <= 3:
                if docs:
                    yield f"Found {len(docs)} matching Sigma rules:\n\n"
                    for idx, rule in enumerate(docs, 1):
                        yield f"Rule {idx}: {rule.get('title','Untitled')}\n```\n"
                        yield self.format_rule(rule)
                        yield "\n```\n"
                else:
                    yield f"No Sigma rules found for: {search_text}\n"
                return

            # If it's more of a question
            question_words = ["what", "how", "why", "when", "where", "who", "explain"]
            if any(query.lower().startswith(w) for w in question_words) or "about" in query.lower():
                llm_prompt = self.create_llm_prompt(query, docs)
                yield from self.get_llm_response(llm_prompt)
            else:
                # Fallback: just treat it as normal LLM prompt
                yield from self.get_llm_response(query)

        except Exception as e:
            self.logger.error(f"Error in pipe: {e}")
            yield f"Error: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run pipeline and return results as list of dicts."""
        output = list(self.pipe(prompt=prompt, **kwargs))
        if not output:
            return []
        return [{"text": "".join(output)}]

if __name__ == "__main__":
    pipeline = Pipeline()
    sample_queries = [
        "search suspicious process creation",
        "what does rule 1 detect?",
        "search_qdrant:powershell",  # triggers prefix-based search
        "list rules for cronjobs"
    ]
    for q in sample_queries:
        print(f"\nQuery: {q}")
        res = pipeline.run(q)
        if res:
            print("Response:", res[0]["text"])
        else:
            print("No response.")
