from typing import List, Dict, Any, Generator
import logging
import json
import os
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

    def __init__(self):
        # Initialize valves with environment variables or defaults
        self.valves = self.Valves(
            **{
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sigma_rules"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                "SIMILARITY_THRESHOLD": float(os.getenv("SIMILARITY_THRESHOLD", 0.3)),
                "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 20))
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

    async def on_startup(self):
        """Verify connections and model on startup."""
        try:
            # Check Qdrant connection
            collection_info = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
            print(f"Connected to Qdrant collection: {collection_info}")
            
            # Verify embedding dimensions match
            test_embedding = self.model.encode(["Test"])[0]
            collection_config = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
            vector_size = collection_config.config.params.vectors.size
            
            if len(test_embedding) != vector_size:
                raise ValueError(f"Model embedding size ({len(test_embedding)}) does not match collection vector size ({vector_size})")
                
        except Exception as e:
            print(f"Startup error: {e}")
            raise
            
    # Add this to check what's in your collection
    def verify_collection(self):
        """Verify collection contents."""
        try:
            count = self.qdrant.count(collection_name=self.valves.QDRANT_COLLECTION)
            print(f"Total points in collection: {count}")
        
            # Get a sample point
            result = self.qdrant.scroll(
                collection_name=self.valves.QDRANT_COLLECTION,
                limit=1,
                with_payload=True,
                with_vectors=True
            )
            if result and result[0]:
                print("Sample point vector dimensions:", len(result[0][0].vector['default']))
                print("Sample point payload keys:", result[0][0].payload.keys())
            return count.count
        except Exception as e:
            print(f"Verification error: {e}")
            return 0
            
    def search_qdrant(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for rules using vector similarity."""
        try:
            # Create a mock rule structure to match ingestion format
            mock_rule = {
                "description": query,
                "title": query,
                "detection": {"keywords": [query]}
            }
            
            # Convert to JSON string to match ingestion format
            query_text = json.dumps(mock_rule, indent=2, default=str)
            
            # Generate embedding for the search query
            query_vector = self.model.encode([query_text])[0].tolist()
            
            # Perform vector similarity search
            results = self.qdrant.search(
                collection_name=self.valves.QDRANT_COLLECTION,
                query_vector={"default": query_vector},
                limit=limit,
                score_threshold=self.valves.SIMILARITY_THRESHOLD
            )
            
            # Extract and format results
            matches = []
            for result in results:
                rule = result.payload
                rule['similarity_score'] = result.score
                matches.append(rule)
                
            return matches

        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def format_rule(self, rule: Dict) -> str:
        """Format rule in Sigma YAML with similarity score."""
        yaml_output = []
        
        # Add similarity score if present
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

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and return vector search results."""
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            # Perform vector similarity search
            matches = self.search_qdrant(query)
            
            if matches:
                yield f"Found {len(matches)} semantically similar Sigma rules:\n\n"
                for idx, rule in enumerate(matches, 1):
                    # Rule number, title, and similarity score
                    yield f"Rule {idx}: {rule.get('title', 'Untitled')} "
                    yield f"(Similarity: {rule.get('similarity_score', 0):.4f})\n"
                    # Rule content in code block
                    yield "```yaml\n"
                    yield self.format_rule(rule)
                    yield "\n```\n\n"
            else:
                yield f"No semantically similar Sigma rules found for query: {query}\n"

        except Exception as e:
            yield f"Error during vector search: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run pipeline and return results."""
        results = list(self.pipe(prompt=prompt, **kwargs))
        return [{"text": "".join(results)}]
