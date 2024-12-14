import logging
import sys
import json
import re
from typing import List, Dict, Any, Generator, Union
from pydantic import BaseModel
import os
import torch
import requests

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    logger.info("Successfully imported required packages")
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    raise

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        EMBEDDING_MODEL_NAME: str
        LLAMA_MODEL_NAME: str
        LLAMA_BASE_URL: str

    def __init__(self) -> None:
        logger.info("Initializing Pipeline")
        try:
            # Valves Configuration
            self.valves = self.Valves(
                **{
                    "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                    "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                    "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-MiniLM-L3-v2"),
                    "LLAMA_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                    "LLAMA_BASE_URL": os.getenv("LLAMA_BASE_URL", "http://ollama:11434"),
                }
            )

            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=self.valves.QDRANT_HOST, 
                port=self.valves.QDRANT_PORT,
                timeout=10.0
            )
            
            # Verify Qdrant connection and collection
            collection_info = self.qdrant_client.get_collection('sigma_rules')
            logger.info(f"Connected to Qdrant. Collection info: {collection_info}")

            # Initialize embedding model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SentenceTransformer(self.valves.EMBEDDING_MODEL_NAME)
            self.model.to(self.device)
            logger.info(f"Model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def search_rules(self, search_terms: List[str]) -> List[Dict]:
        """Search for rules using exact phrase matching."""
        logger.info(f"Starting search with terms: {search_terms}")
        matches = []

        try:
            # Get all points from Qdrant
            scroll_result = self.qdrant_client.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"Retrieved {len(scroll_result[0])} records from Qdrant")

            # Search through each record
            for point in scroll_result[0]:
                payload = point.payload
                logger.debug(f"Checking rule: {payload.get('title')}")

                for term in search_terms:
                    term_lower = term.lower()
                    
                    # Check title
                    if payload.get('title') and term_lower in payload['title'].lower():
                        logger.debug(f"Found match in title for term: {term}")
                        matches.append(payload)
                        break
                        
                    # Check description
                    elif payload.get('description') and term_lower in payload['description'].lower():
                        logger.debug(f"Found match in description for term: {term}")
                        matches.append(payload)
                        break
                        
                    # Check detection rules
                    elif payload.get('detection') and term_lower in json.dumps(payload['detection']).lower():
                        logger.debug(f"Found match in detection rules for term: {term}")
                        matches.append(payload)
                        break

            logger.info(f"Found {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"Error in search_rules: {e}", exc_info=True)
            return []

    def format_rule_output(self, payload: Dict) -> str:
        """Format a Sigma rule payload for output."""
        result = []

        if payload.get('title'):
            result.append(f"Title: {payload['title']}")
        if payload.get('description'):
            result.append(f"Description: {payload['description']}")
        if payload.get('detection'):
            result.append("Detection Rules:")
            result.append(json.dumps(payload['detection'], indent=2))

        return '\n'.join(result)

    def generate_llm_response(self, query: str) -> str:
        """Send the query to the Ollama server for a response."""
        logger.info(f"Generating LLM response for: {query}")
        try:
            response = requests.post(
                url=f"{self.valves.LLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.valves.LLAMA_MODEL_NAME,
                    "prompt": query,
                    "options": {
                        "seed": 123,
                        "temperature": 0
                    },
                },
                stream=True
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        full_response += chunk
                    except json.JSONDecodeError:
                        continue

            return full_response.strip() or "No response generated."

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and yield results."""
        try:
            query = prompt or kwargs.get('user_message', '')
            if not query:
                return

            # Check for quoted terms
            search_terms = re.findall(r'"([^"]*)"', query)
            
            if search_terms:
                logger.info(f"Found search terms: {search_terms}")
                matches = self.search_rules(search_terms)
                
                if matches:
                    yield f"Found {len(matches)} Sigma rules:\n\n"
                    for idx, match in enumerate(matches, 1):
                        yield f"Rule {idx}:\n"
                        yield self.format_rule_output(match)
                        yield "\n" + "-"*80 + "\n\n"
                else:
                    yield f"No Sigma rules found matching: {', '.join(search_terms)}\n"
            else:
                # Use LLM for non-search queries
                result = self.generate_llm_response(query)
                yield result

        except Exception as e:
            logger.error(f"Error in pipe method: {e}", exc_info=True)
            yield f"Error: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Process input and return results."""
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            return [{"text": "".join(results)}]
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)
            return [{"error": str(e)}]
