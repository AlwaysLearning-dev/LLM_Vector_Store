from typing import List, Dict, Any, Generator
import logging
import json
import re
import os
import requests
from qdrant_client import QdrantClient
from pydantic import BaseModel

# --- Include LangChain imports ---
from langchain.schema import Document

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_COLLECTION: str
        LLM_MODEL_NAME: str
        LLM_BASE_URL: str
        ENABLE_CONTEXT: bool
        LOG_LEVEL: str

    def __init__(self):
        self.valves = self.Valves(
            QDRANT_HOST=os.getenv("QDRANT_HOST", "qdrant"),
            QDRANT_PORT=int(os.getenv("QDRANT_PORT", 6333)),
            QDRANT_COLLECTION=os.getenv("QDRANT_COLLECTION", "sigma_rules"),
            LLM_MODEL_NAME=os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
            LLM_BASE_URL=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            ENABLE_CONTEXT=os.getenv("ENABLE_CONTEXT", "true").lower() == "true",
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO")
        )
        self.qdrant = QdrantClient(host=self.valves.QDRANT_HOST, port=self.valves.QDRANT_PORT)

    async def on_startup(self):
        try:
            _ = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise

    async def on_shutdown(self):
        pass

    def extract_search_terms(self, query: str) -> List[str]:
        phrases = re.findall(r'"([^"]*)"', query)
        if phrases: return phrases
        terms = [t.strip() for t in query.split() if t.strip()]
        return terms

    def search_qdrant(self, terms: List[str]) -> List[Dict]:
        try:
            result = self.qdrant.scroll(
                collection_name=self.valves.QDRANT_COLLECTION,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            matches = []
            if result and result[0]:
                for point in result[0]:
                    payload = point.payload
                    searchable = json.dumps(payload, default=str).lower()
                    if any(term.lower() in searchable for term in terms):
                        matches.append(payload)
            return matches
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    def get_context_from_rules(self, rules: List[Dict]) -> str:
        context_lines = []
        for i, r in enumerate(rules, start=1):
            context_lines.append(f"Rule {i}: {r.get('title','Untitled')}")
            if r.get('description'):
                context_lines.append(r['description'])
        return "\n".join(context_lines)

    def create_llm_prompt(self, query: str, context: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Query:\n{query}\n\n"
            "Answer using these rules whenever possible."
        )

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        search_terms = self.extract_search_terms(query)
        matches = self.search_qdrant(search_terms) if search_terms else []

        # Build prompt
        if self.valves.ENABLE_CONTEXT and matches:
            context = self.get_context_from_rules(matches)
            llm_prompt = self.create_llm_prompt(query, context)
        else:
            llm_prompt = query

        try:
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
                    except:
                        continue
        except Exception as e:
            yield f"Error: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        results = list(self.pipe(prompt=prompt, **kwargs))
        return [{"text": "".join(results)}] if results else []
