from typing import List, Dict, Any, Generator
import logging
import json
import re
import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointSearchParams
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_COLLECTION: str
        LLM_MODEL_NAME: str
        LLM_BASE_URL: str
        EMBEDDING_MODEL_URL: str
        ENABLE_CONTEXT: bool
        LOG_LEVEL: str

    def __init__(self):
        # Initialize valves with environment variables or defaults
        self.valves = self.Valves(
            **{
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sigma_rules"),
                "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME", "llama3.2"),
                "LLM_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                "EMBEDDING_MODEL_URL": os.getenv("EMBEDDING_MODEL_URL", "http://embedding-model:8000"),
                "ENABLE_CONTEXT": os.getenv("ENABLE_CONTEXT", "true").lower() == "true",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            }
        )

        logging.basicConfig(level=self.valves.LOG_LEVEL)
        self.qdrant = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the query text."""
        try:
            response = requests.post(self.valves.EMBEDDING_MODEL_URL, json={"text": text})
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            if not embedding:
                raise ValueError("Empty embedding returned.")
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise

    def search_qdrant_vector(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """Search Qdrant using vector similarity."""
        try:
            search_result = self.qdrant.search(
                collection_name=self.valves.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            return [result.payload for result in search_result]
        except Exception as e:
            logging.error(f"Vector search error: {e}")
            return []


    def search_qdrant_keywords(self, terms: List[str]) -> List[Dict]:
        """Fallback to keyword search."""
        try:
            scroll_result = self.qdrant.scroll(
                collection_name=self.valves.QDRANT_COLLECTION,
                limit=100,
                with_payload=True
            )
            matches = []
            for point in scroll_result[0]:
                payload = point.payload
                searchable_text = ' '.join(
                    [str(v).lower() for v in payload.values() if isinstance(v, (str, list, dict))]
                )
                if any(term.lower() in searchable_text for term in terms):
                    matches.append(payload)
            return matches
        except Exception as e:
            logging.error(f"Keyword search error: {e}")
            return []

    def search_qdrant(self, query: str) -> List[Dict]:
        """Search Qdrant with vector embeddings first, fallback to keywords."""
        try:
            query_vector = self.get_embedding(query)
            logging.info("Performing vector search.")
            results = self.search_qdrant_vector(query_vector)
            if results:
                return results
        except Exception as e:
            logging.warning(f"Vector search failed: {e}. Falling back to keywords.")
        terms = self.extract_search_terms(query)
        return self.search_qdrant_keywords(terms)

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract potential keywords or phrases from query."""
        phrases = re.findall(r'"([^"]*)"', query)
        if not phrases:
            return [term.strip() for term in query.split()]
        return phrases

    def format_rule(self, rule: Dict) -> str:
        """Format Sigma rule into YAML-like format."""
        yaml_output = []
        for field, value in rule.items():
            if isinstance(value, dict):
                yaml_output.append(f"{field}:")
                for key, val in value.items():
                    yaml_output.append(f"  {key}: {val}")
            elif isinstance(value, list):
                yaml_output.append(f"{field}:")
                for item in value:
                    yaml_output.append(f"  - {item}")
            else:
                yaml_output.append(f"{field}: {value}")
        return "\n".join(yaml_output)

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Pipeline logic to process queries and return results."""
        query = prompt or kwargs.get("user_message", "")
        if not query:
            yield "No query provided."
            return
        try:
            # Search Qdrant for results
            results = self.search_qdrant(query)
            if results:
                yield f"Found {len(results)} matching Sigma rules:\n\n"
                for idx, rule in enumerate(results, 1):
                    yield f"Rule {idx}: {rule.get('title', 'Untitled')}\n"
                    yield "```yaml\n"
                    yield self.format_rule(rule)
                    yield "\n```\n\n"
            else:
                yield f"No matches found for query: {query}"
        except Exception as e:
            yield f"Error processing query: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run the pipeline and return results."""
        return [{"text": "".join(self.pipe(prompt=prompt, **kwargs))}]
