from typing import List, Dict, Any, Generator
import logging
import json
import os
import requests
from qdrant_client import QdrantClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_COLLECTION: str
        EMBEDDING_MODEL: str
        SIMILARITY_THRESHOLD: float
        BATCH_SIZE: int
        LLM_MODEL_NAME: str
        LLM_BASE_URL: str
        ENABLE_CONTEXT: bool

    def __init__(self):
        # Initialize valves with environment variables or defaults
        self.valves = self.Valves(
            **{
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sigma_rules"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", 0.3)),
                "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 20)),
                "LLM_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama2"),
                "LLM_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                "ENABLE_CONTEXT": os.getenv("ENABLE_CONTEXT", "true").lower() == "true"
            }
        )

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )
        
        # Initialize the embedding model
        self.model = SentenceTransformer(self.valves.EMBEDDING_MODEL)
        print(f"Initialized embedding model: {self.valves.EMBEDDING_MODEL}")

    def verify_collection(self):
        """Verify collection contents."""
        try:
            count = self.qdrant.count(collection_name=self.valves.QDRANT_COLLECTION)
            print(f"Total points in collection: {count.count}")
            
            result = self.qdrant.scroll(
                collection_name=self.valves.QDRANT_COLLECTION,
                limit=1,
                with_payload=True,
                with_vectors=True
            )
            if result and result[0]:
                print("Sample point vector dimensions:", len(result[0][0].vector['default']))
                print("Sample point payload keys:", list(result[0][0].payload.keys()))
            return count.count
        except Exception as e:
            print(f"Verification error: {e}")
            return 0

    async def on_startup(self):
        """Verify connections and model on startup."""
        try:
            # Check Qdrant connection
            collection_info = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
            print(f"Connected to Qdrant collection: {collection_info}")
            
            # Verify collection contents
            count = self.verify_collection()
            print(f"Collection contains {count} rules")
            
            # Test LLM connection
            try:
                response = requests.get(f"{self.valves.LLM_BASE_URL}/api/version")
                print(f"Connected to LLM service: {response.json()}")
            except Exception as e:
                print(f"Warning: LLM service not available: {e}")
            
        except Exception as e:
            print(f"Startup error: {e}")
            raise

    def search_qdrant(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for rules using vector similarity."""
        try:
            mock_rule = {
                "description": query,
                "title": query,
                "detection": {"keywords": [query]}
            }
            
            query_text = json.dumps(mock_rule, indent=2, default=str)
            query_vector = self.model.encode([query_text])[0].tolist()
            
            results = self.qdrant.search(
                collection_name=self.valves.QDRANT_COLLECTION,
                query_vector={"default": query_vector},
                limit=limit,
                score_threshold=self.valves.SIMILARITY_THRESHOLD
            )
            
            matches = []
            for result in results:
                rule = result.payload
                rule['similarity_score'] = result.score
                matches.append(rule)
                
            return matches

        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def get_context_from_rules(self, rules: List[Dict]) -> str:
        """Create context string from rules for LLM."""
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

    def format_rule(self, rule: Dict) -> str:
        """Format rule in Sigma YAML with similarity score."""
        yaml_output = []
        
        if 'similarity_score' in rule:
            yaml_output.append(f"similarity_score: {rule['similarity_score']:.4f}")
            yaml_output.append("---")
        
        fields = [
            'title', 'id', 'status', 'description', 'references', 'author',
            'date', 'modified', 'tags', 'logsource', 'detection',
            'falsepositives', 'level', 'filename'
        ]
        
        for field in fields:
            value = rule.get(field)
            if value is not None:
                if isinstance(value, (dict, list)):
                    yaml_output.append(f"{field}:")
                    value_str = json.dumps(value, indent=2)
                    yaml_output.extend(f"  {line}" for line in value_str.split('\n'))
                else:
                    yaml_output.append(f"{field}: {value}")
                    
        return '\n'.join(yaml_output)

    def looks_like_question(self, query: str) -> bool:
        """Check if query is a question about rules."""
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain']
        contains_about = 'about' in query.lower()
        starts_with_question = any(query.lower().startswith(word) for word in question_words)
        return starts_with_question or contains_about

    def create_llm_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM that includes context."""
        return f"""Here are some relevant Sigma detection rules for context:

{context}

Based on these rules, please answer this question:
{query}

Please be specific and refer to the rules when applicable."""

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and return both vector search results and LLM analysis."""
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            # Perform vector similarity search
            matches = self.search_qdrant(query)
            
            # If it's a question and we have matches, use LLM
            if self.looks_like_question(query) and matches and self.valves.ENABLE_CONTEXT:
                context = self.get_context_from_rules(matches)
                llm_prompt = self.create_llm_prompt(query, context)
                
                # Get LLM response
                response = requests.post(
                    url=f"{self.valves.LLM_BASE_URL}/api/generate",
                    json={"model": self.valves.LLM_MODEL_NAME, "prompt": llm_prompt},
                    stream=True
                )
                
                yield "LLM Analysis:\n"
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = json.loads(line)
                            yield data.get("response", "")
                        except json.JSONDecodeError:
                            continue
                yield "\n\nRelevant Rules:\n"
            
            # Output matching rules
            if matches:
                yield f"Found {len(matches)} semantically similar Sigma rules:\n\n"
                for idx, rule in enumerate(matches, 1):
                    yield f"Rule {idx}: {rule.get('title', 'Untitled')} "
                    yield f"(Similarity: {rule.get('similarity_score', 0):.4f})\n"
                    yield "```yaml\n"
                    yield self.format_rule(rule)
                    yield "\n```\n\n"
            else:
                yield f"No semantically similar Sigma rules found for query: {query}\n"

        except Exception as e:
            yield f"Error during search: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run pipeline and return results."""
        results = list(self.pipe(prompt=prompt, **kwargs))
        return [{"text": "".join(results)}]
