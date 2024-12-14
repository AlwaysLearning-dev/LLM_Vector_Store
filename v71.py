import logging
import sys
import json
import re
from typing import List, Dict, Any, Generator, Union
from pydantic import BaseModel
import os
import requests
from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class Pipeline:

    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        LLAMA_MODEL_NAME: str
        LLAMA_BASE_URL: str

    def __init__(self):
        """Initialize the pipeline with required components."""
        logger.info("Initializing Pipeline")
        try:
            # Valves Configuration
            self.valves = self.Valves(
                **{
                    "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                    "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                    "LLAMA_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                    "LLAMA_BASE_URL": os.getenv("LLAMA_BASE_URL", "http://ollama:11434")
                }
            )

            # Initialize Qdrant client
            self.qdrant = QdrantClient(
                host=self.valves.QDRANT_HOST,
                port=self.valves.QDRANT_PORT
            )
            logger.info("Qdrant client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from query."""
        phrases = re.findall(r'"([^"]*)"', query)
        if not phrases:
            about_match = re.search(r'about\s+(\w+)', query.lower())
            related_match = re.search(r'related to\s+(\w+)', query.lower())
            if about_match:
                return [about_match.group(1)]
            if related_match:
                return [related_match.group(1)]
            return query.split()
        return phrases

    def search_qdrant(self, terms: List[str]) -> List[Dict]:
        """Search for rules matching terms."""
        try:
            result = self.qdrant.scroll(
                collection_name="sigma_rules",
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            matches = set()
            if result and result[0]:
                for point in result[0]:
                    payload = point.payload
                    searchable_text = ' '.join(str(v) for v in payload.values()).lower()

                    for term in terms:
                        if term.lower() in searchable_text:
                            matches.add((payload.get('title', ''), json.dumps(payload)))
                            break

            return [json.loads(match[1]) for match in matches]

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}", exc_info=True)
            return []

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
            if value is not None:
                if field == 'description':
                    yaml_output.append(f"description: |")
                    for line in str(value).split('\n'):
                        yaml_output.append(f"    {line}")
                elif isinstance(value, (list, dict)):
                    yaml_output.append(f"{field}:")
                    if isinstance(value, list):
                        for item in value:
                            yaml_output.append(f"    - {item}")
                    else:
                        yaml_output.extend(f"    {line}" for line in json.dumps(value, indent=2).split('\n'))
                else:
                    yaml_output.append(f"{field}: {value}")

        return '\n'.join(yaml_output)

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
            search_terms = self.extract_search_terms(query)
            matches = self.search_qdrant(search_terms) if search_terms else []

            if matches:
                yield '<div style="font-size: 8pt;">'
                yield f"Found {len(matches)} matching Sigma rules:\n\n"
                for idx, rule in enumerate(matches, 1):
                    yield f"### Rule {idx}\n"
                    yield "```yaml\n"
                    yield self.format_rule(rule)
                    yield "\n```\n\n"
                yield "</div>"
                return

            context = '\n\n'.join(self.format_rule(match) for match in matches) if matches else ''
            llm_prompt = self.create_llm_prompt(query, context) if context else query

            response = requests.post(
                url=f"{self.valves.LLAMA_BASE_URL}/api/generate",
                json={"model": self.valves.LLAMA_MODEL_NAME, "prompt": llm_prompt},
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
            logger.error(f"Error in pipe method: {e}", exc_info=True)
            yield f"Error: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run pipeline and return results."""
        results = list(self.pipe(prompt=prompt, **kwargs))
        if not results:
            return []
        return [{"text": "".join(results)}]
