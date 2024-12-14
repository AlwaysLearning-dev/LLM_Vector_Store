from typing import List, Dict, Any, Generator, Union
from pydantic import BaseModel
import logging
import sys
import json
import re
import os
import requests
from qdrant_client import QdrantClient

# Force DEBUG logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Create a StreamHandler for stdout if none exists
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# Create a file handler for persistent logs
file_handler = logging.FileHandler('/tmp/pipeline_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        LLAMA_MODEL_NAME: str
        LLAMA_BASE_URL: str

    def __init__(self) -> None:
        logger.info("=== Pipeline Initialization Starting ===")
        try:
            # Test logging
            logger.debug("Debug logging test")
            logger.info("Info logging test")
            
            # Valves Configuration
            self.valves = self.Valves(
                **{
                    "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                    "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                    "LLAMA_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                    "LLAMA_BASE_URL": os.getenv("LLAMA_BASE_URL", "http://ollama:11434"),
                }
            )
            logger.debug(f"Configuration loaded: {self.valves}")

            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=self.valves.QDRANT_HOST, 
                port=self.valves.QDRANT_PORT,
                timeout=10.0
            )
            
            # Test Qdrant connection
            collection_info = self.qdrant_client.get_collection('sigma_rules')
            logger.debug(f"Qdrant connection successful. Collection info: {collection_info}")
            
            # Write test message to file
            with open('/tmp/pipeline_init.log', 'w') as f:
                f.write("Pipeline initialized successfully\n")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            # Write error to file
            with open('/tmp/pipeline_error.log', 'w') as f:
                f.write(f"Pipeline initialization failed: {str(e)}\n")
            raise

    def search_qdrant(self, term: str) -> List[Dict]:
        logger.debug(f"=== Starting Qdrant search for term: {term} ===")
        try:
            # Get records from Qdrant
            scroll_result = self.qdrant_client.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result or not scroll_result[0]:
                logger.warning("No records returned from Qdrant")
                return []
            
            logger.debug(f"Retrieved {len(scroll_result[0])} records from Qdrant")
            
            # Search for matches
            matches = []
            for point in scroll_result[0]:
                payload = point.payload
                logger.debug(f"Checking rule: {payload.get('title', 'No title')}")
                
                # Search in all relevant fields
                searchable_text = ' '.join(str(v) for v in [
                    payload.get('title', ''),
                    payload.get('description', ''),
                    json.dumps(payload.get('detection', {}))
                ]).lower()
                
                if term.lower() in searchable_text:
                    logger.debug(f"Found match: {payload.get('title')}")
                    matches.append(payload)
            
            logger.debug(f"Found {len(matches)} matches for term: {term}")
            
            # Write search results to file
            with open('/tmp/search_results.log', 'a') as f:
                f.write(f"\nSearch for '{term}' found {len(matches)} matches\n")
                for match in matches:
                    f.write(f"- {match.get('title', 'No title')}\n")
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}", exc_info=True)
            return []

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        logger.debug("=== Starting pipe method ===")
        try:
            # Get query
            query = prompt or kwargs.get('user_message', '')
            logger.debug(f"Processing query: {query}")
            
            # Write query to file
            with open('/tmp/queries.log', 'a') as f:
                f.write(f"\nProcessing query: {query}\n")
            
            # Look for quoted terms
            if '"' in query:
                search_terms = re.findall(r'"([^"]*)"', query)
                logger.debug(f"Found search terms: {search_terms}")
                
                if search_terms:
                    # Do Qdrant search
                    logger.debug("Starting Qdrant search")
                    all_matches = []
                    for term in search_terms:
                        matches = self.search_qdrant(term)
                        all_matches.extend(matches)
                    
                    if all_matches:
                        yield f"Found {len(all_matches)} Sigma rules matching your search:\n\n"
                        for idx, match in enumerate(all_matches, 1):
                            result = f"Rule {idx}:\n"
                            if match.get('title'):
                                result += f"Title: {match['title']}\n"
                            if match.get('description'):
                                result += f"Description: {match['description']}\n"
                            if match.get('detection'):
                                result += f"Detection:\n{json.dumps(match['detection'], indent=2)}\n"
                            result += "\n" + "-"*80 + "\n\n"
                            logger.debug(f"Yielding result: {result[:100]}...")  # Log first 100 chars
                            yield result
                    else:
                        msg = f"No Sigma rules found matching: {', '.join(search_terms)}\n"
                        logger.debug(f"No matches found: {msg}")
                        yield msg
                    return
            
            # If no quotes or no matches, use LLM
            logger.debug("Using LLM for response")
            response = requests.post(
                url=f"{self.valves.LLAMA_BASE_URL}/api/generate",
                json={"model": self.valves.LLAMA_MODEL_NAME, "prompt": query},
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        yield data.get("response", "")
                    except json.JSONDecodeError:
                        continue
                
        except Exception as e:
            error_msg = f"Error in pipe method: {str(e)}"
            logger.error(error_msg, exc_info=True)
            with open('/tmp/pipeline_errors.log', 'a') as f:
                f.write(f"\n{error_msg}\n")
            yield error_msg

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        logger.debug("=== Starting run method ===")
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            if not results:
                logger.warning("No results generated")
                return []
            
            logger.debug("Results generated successfully")
            return [{"text": "".join(results)}]
            
        except Exception as e:
            error_msg = f"Error in run method: {str(e)}"
            logger.error(error_msg, exc_info=True)
            with open('/tmp/pipeline_errors.log', 'a') as f:
                f.write(f"\n{error_msg}\n")
            return [{"error": error_msg}]
