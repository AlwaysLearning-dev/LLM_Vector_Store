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
        logger.info(f"Extracting search terms from query: {query}")
        phrases = re.findall(r'"([^"]*)"', query)
        logger.debug(f"Extracted quoted phrases: {phrases}")
        return phrases

    def search_rules(self, search_terms: List[str]) -> List[Dict]:
        """Search for rules using exact phrase matching."""
        logger.info(f"Starting search with terms: {search_terms}")
        matches = set()

        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            for point in scroll_result[0]:
                payload = point.payload

                for term in search_terms:
                    term_lower = term.lower()
                    found = False

                    # Adjust matching logic for short terms
                    if len(term.split()) <= 3:  # For 1-3 tokens, strict match
                        if payload.get('title') and payload['title'].strip().lower() == term_lower:
                            found = True
                        elif payload.get('description') and payload['description'].strip().lower() == term_lower:
                            found = True
                    else:  # For longer queries, partial matching
                        if payload.get('title') and term_lower in payload['title'].lower():
                            found = True
                        elif payload.get('description') and term_lower in payload['description'].lower():
                            found = True
                        elif payload.get('detection') and term_lower in json.dumps(payload['detection']).lower():
                            found = True

                    if found:
                        matches.add((payload.get('title', ''), json.dumps(payload, sort_keys=True)))
                        break

            unique_matches = [json.loads(match[1]) for match in matches]
            logger.info(f"Found {len(unique_matches)} unique matches")
            return unique_matches

        except Exception as e:
            logger.error(f"Error in search_rules: {e}", exc_info=True)
            return []

    def format_rule_output(self, payload: Dict) -> str:
        """Format a Sigma rule payload for output."""
        logger.info(f"Formatting rule output for payload: {payload.get('title', 'No Title')}")
        result = []

        if payload.get('title'):
            result.append(f"Title: {payload['title']}")
        if payload.get('description'):
            result.append(f"Description: {payload['description']}")
        if payload.get('detection'):
            result.append("Detection Rules:")
            result.append(json.dumps(payload['detection'], indent=2))
        if payload.get('filename'):
            result.append(f"Filename: {payload['filename']}")

        formatted_result = '\n'.join(result)
        logger.debug(f"Formatted rule output: {formatted_result}")
        return formatted_result

    def generate_llm_response(self, query: str, context: str = "") -> str:
        """Send the query to the Ollama server for a response."""
        logger.info(f"Generating LLM response for query: {query}")
        try:
            full_query = query
            if context and len(query.split()) > 3:  # Add context only for sufficiently descriptive queries
                full_query = f"Consider the following Sigma rules from the Qdrant database while answering: {context}\n\n{query}"

            response = requests.post(
                url=f"{self.valves.LLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.valves.LLAMA_MODEL_NAME,
                    "prompt": full_query,
                    "options": {
                        "seed": 123,
                        "temperature": 0
                    },
                },
                stream=True  # Enable streaming
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        full_response += chunk
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping non-JSON line: {line} | Error: {e}")

            if not full_response.strip():
                logger.warning("LLM response is empty")
                return "The model did not generate a response. Please try a different query."

            logger.info("LLM response generated successfully")
            return full_response.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request Error generating LLM response: {e}", exc_info=True)
            return "An error occurred while generating the response due to a network issue."

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return "An unexpected error occurred while generating the response."

    async def on_startup(self):
        """Initialize resources on startup."""
        logger.info("Pipeline startup: Resources ready.")

    async def on_shutdown(self):
        """Clean up resources on shutdown."""
        logger.info("Pipeline shutdown: Resources released.")

    def pipe(self, prompt: str = None, **kwargs) -> Union[str, Generator, None]:
        """OpenWebUI compatible pipe method that yields results as a stream."""
        logger.info(f"Pipe method called with prompt: {prompt}")
        try:
            query = prompt or kwargs.get('user_message', '')
            if not query:
                raise ValueError("No query provided")

            logger.info(f"Processing query: {query}")

            search_terms = self.extract_search_terms(query)
            logger.info(f"Extracted search terms: {search_terms}")

            if search_terms:
                logger.info(f"Searching Qdrant with terms: {search_terms}")
                matches = self.search_rules(search_terms)
                logger.info(f"Search results: {matches}")

                if matches:
                    context = "\n\n".join([self.format_rule_output(match) for match in matches])
                    logger.debug(f"Context for LLM: {context}")
                    yield f"Found {len(matches)} Sigma rules matching your query:\n\n"
                    for idx, match in enumerate(matches, 1):
                        result_text = f"Match {idx}:\n"
                        result_text += self.format_rule_output(match)
                        result_text += "\n\n" + "-" * 80 + "\n\n"
                        yield result_text
                    yield self.generate_llm_response(query, context)
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
            logger.debug(f"Run method results: {results}")
            return [{"text": "".join(results)}]
        except Exception as e:
            logger.error(f"Error in run method: {e}", exc_info=True)
            return [{"error": str(e)}]
