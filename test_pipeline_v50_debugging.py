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
        """Initialize the pipeline with required components."""
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
            logger.info("Qdrant client initialized")

            # Initialize small, efficient model on GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SentenceTransformer(self.valves.EMBEDDING_MODEL_NAME)
            self.model.to(self.device)
            logger.info(f"Model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract only phrases enclosed in quotes."""
        phrases = re.findall(r'"([^"]*)"', query)
        logger.debug(f"Extracted quoted phrases: {phrases}")
        return phrases

    def search_rules(self, search_terms: List[str]) -> List[Dict]:
        """Search for rules using exact phrase matching."""
        logger.debug(f"Searching for terms: {search_terms}")
        matches = set()

        # Get points
        scroll_result = self.qdrant_client.scroll(
            collection_name="sigma_rules",
            limit=100,
            with_payload=True,
            with_vectors=False
        )

        for point in scroll_result[0]:
            payload = point.payload

            # Check each search term
            for term in search_terms:
                term_lower = term.lower()
                found = False

                # Search in title and description first
                if payload.get('title') and term_lower in payload['title'].lower():
                    found = True
                elif payload.get('description') and term_lower in payload['description'].lower():
                    found = True
                # Only search other fields if not found in title/description
                elif payload.get('detection') and term_lower in json.dumps(payload['detection']).lower():
                    found = True

                if found:
                    matches.add((payload.get('title', ''), json.dumps(payload, sort_keys=True)))
                    break

        unique_matches = [json.loads(match[1]) for match in matches]
        logger.debug(f"Found {len(unique_matches)} unique matches")
        return unique_matches

    def format_rule_output(self, payload: Dict) -> str:
        """Format a Sigma rule payload for output."""
        result = []

        # Add title and description
        if payload.get('title'):
            result.append(f"Title: {payload['title']}")
        if payload.get('description'):
            result.append(f"Description: {payload['description']}")

        # Add detection rules
        if payload.get('detection'):
            result.append("Detection Rules:")
            result.append(json.dumps(payload['detection'], indent=2))

        # Add filename
        if payload.get('filename'):
            result.append(f"Filename: {payload['filename']}")

        return '\n'.join(result)

    def generate_llm_response(self, query: str) -> str:
        """Send the query to the Ollama server for a response."""
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
                stream=True  # Enable streaming
            )
            response.raise_for_status()

            # Collect streamed chunks of the response
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Skip empty lines
                    try:
                        # Parse each line as JSON
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        full_response += chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON line in response: {line}")

            if not full_response.strip():
                return "The model did not generate a response. Please try a different query."
            return full_response.strip()

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return "An error occurred while generating the response."

    async def on_startup(self):
        """Initialize resources on startup."""
        logger.info("Pipeline startup: Resources ready.")

    async def on_shutdown(self):
        """Clean up resources on shutdown."""
        logger.info("Pipeline shutdown: Resources released.")

    def pipe(self, prompt: str = None, **kwargs) -> Union[str, Generator, None]:
        """OpenWebUI compatible pipe method that yields results as a stream."""
        try:
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided")

            logger.info(f"Processing query: {query}")

            # Extract quoted search terms
            search_terms = self.extract_search_terms(query)
            logger.info(f"Extracted search terms: {search_terms}")

            if search_terms:  # PRIORITIZE SEARCH QUERIES
                matches = self.search_rules(search_terms)
                logger.info(f"Found {len(matches)} matches in Qdrant.")

                if matches:
                    yield f"Found {len(matches)} Sigma rules matching your query:\n\n"
                    for idx, match in enumerate(matches, 1):
                        result_text = f"Match {idx}:\n"
                        result_text += self.format_rule_output(match)
                        result_text += "\n\n" + "-" * 80 + "\n\n"
                        yield result_text
                else:
                    yield "No matches found for the quoted terms.\n"
            else:
                logger.info("No quoted terms; generating default LLM response.")
                yield self.generate_llm_response(query)

        except Exception as e:
            logger.error(f"Error in pipe method: {e}", exc_info=True)
            yield f"Error processing query: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Legacy method for non-streaming results."""
        logger.info(f"Run method called with prompt: {prompt}")
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            return [{"text": "".join(results)}]
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)
            return [{"error": str(e)}]
