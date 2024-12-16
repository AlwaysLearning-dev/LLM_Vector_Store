"""
title: Enhanced Qdrant Sigma Rules Pipeline with Context Awareness
author: open-webui
date: 2024-12-14
version: 1.2
license: MIT
description: Advanced pipeline for searching and analyzing Sigma rules with enhanced context
requirements: 
- qdrant-client
- requests
- sentence-transformers
"""

from typing import List, Dict, Any, Generator
import logging
import json
import re
import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_COLLECTION: str
        LLM_MODEL_NAME: str
        LLM_BASE_URL: str
        ENABLE_CONTEXT: bool
        LOG_LEVEL: str
        SEARCH_PREFIX: str
        SEMANTIC_SEARCH_TOP_K: int

    def __init__(self):
        # Initialize valves with environment variables or defaults
        self.valves = self.Valves(
            **{
                "QDRANT_HOST": os.getenv("QDRANT_HOST", "qdrant"),
                "QDRANT_PORT": int(os.getenv("QDRANT_PORT", 6333)),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sigma_rules"),
                "LLM_MODEL_NAME": os.getenv("LLAMA_MODEL_NAME", "llama3.2"),
                "LLM_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                "ENABLE_CONTEXT": os.getenv("ENABLE_CONTEXT", "true").lower() == "true",
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "SEARCH_PREFIX": os.getenv("SEARCH_PREFIX", "search_qdrant:"),
                "SEMANTIC_SEARCH_TOP_K": int(os.getenv("SEMANTIC_SEARCH_TOP_K", "5"))
            }
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )

        # Generate rule summary for persistent context
        self.rule_summary = self._generate_rule_summary()

    def _generate_rule_summary(self) -> str:
        """Generate a high-level summary of available Sigma rules."""
        try:
            # Fetch all rules (without limit)
            result = self.qdrant.scroll(
                collection_name=self.valves.QDRANT_COLLECTION,
                with_payload=True,
                with_vectors=False
            )

            if not result or not result[0]:
                return "No Sigma rules found in the database."

            categories = {}
            total_rules = len(result[0])
            for point in result[0]:
                payload = point.payload
                for tag in payload.get('tags', []):
                    if tag.startswith('attack.'):
                        category = tag.split('.')[1]
                        categories[category] = categories.get(category, 0) + 1
            
            summary = f"Sigma Rules Database Summary (Total Rules: {total_rules}):\n"
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_rules) * 100
                summary += f"- {category.replace('_', ' ').title()}: {count} rules ({percentage:.1f}%)\n"
            
            return summary

        except Exception as e:
            return f"Error generating rule summary: {e}"

    def _semantic_search(self, query: str) -> List[Dict]:
        """Perform semantic search on Sigma rules."""
        try:
            # Embed the query
            query_embedding = self.embedding_model.encode(query)

            # Perform semantic search
            search_results = self.qdrant.search(
                collection_name=self.valves.QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=self.valves.SEMANTIC_SEARCH_TOP_K,
                with_payload=True
            )

            # Convert results to rule dictionaries
            return [point.payload for point in search_results]

        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms with advanced matching."""
        # Check for custom search prefix
        search_prefix = self.valves.SEARCH_PREFIX
        if query.startswith(search_prefix):
            search_query = query[len(search_prefix):].strip()
            
            # Try exact phrase matching first
            phrases = re.findall(r'"([^"]*)"', search_query)
            if phrases:
                return phrases
            
            # Split remaining query
            return [term.strip() for term in search_query.split() if term.strip()]
        
        # Existing search term extraction
        phrases = re.findall(r'"([^"]*)"', query)
        if not phrases:
            # Check for rule title or specific references
            rule_title_match = re.search(r'Rule\s*"([^"]+)"', query)
            about_match = re.search(r'about\s+"?([^"]+)"?', query.lower())
            related_match = re.search(r'related to\s+"?([^"]+)"?', query.lower())
            
            if rule_title_match:
                return [rule_title_match.group(1)]
            if about_match:
                return [about_match.group(1).strip()]
            if related_match:
                return [related_match.group(1).strip()]
            
            # Standard term extraction
            return [term.strip() for term in query.split() if term.strip()]
        
        return phrases

    def search_qdrant(self, terms: List[str]) -> List[Dict]:
        """Enhanced search across multiple matching strategies."""
        try:
            # Scroll through all rules
            result = self.qdrant.scroll(
                collection_name=self.valves.QDRANT_COLLECTION,
                with_payload=True,
                with_vectors=False
            )

            if not result or not result[0]:
                return []

            matches = set()
            for point in result[0]:
                payload = point.payload
                searchable_parts = []
                
                # Comprehensive field extraction
                search_fields = {
                    'simple': ['title', 'id', 'status', 'author', 
                               'date', 'modified', 'level', 'filename'],
                    'complex': ['description', 'references', 'tags', 'falsepositives'],
                    'nested': ['logsource', 'detection']
                }
                
                # Compile searchable text
                for field in search_fields['simple']:
                    if payload.get(field):
                        searchable_parts.append(str(payload[field]))
                
                for field in search_fields['complex']:
                    field_value = payload.get(field)
                    if field_value:
                        if isinstance(field_value, list):
                            searchable_parts.extend(str(item) for item in field_value)
                        else:
                            searchable_parts.append(str(field_value))
                
                # Handle nested fields
                if payload.get('logsource'):
                    searchable_parts.extend(str(v) for v in payload['logsource'].values())
                
                if payload.get('detection'):
                    detection_str = json.dumps(payload['detection'])
                    searchable_parts.append(detection_str)
                
                # MITRE ATT&CK tag handling
                if payload.get('tags'):
                    attack_tags = [tag for tag in payload['tags'] if tag.startswith('attack.')]
                    searchable_parts.extend(attack_tags)
                
                # Combine searchable parts
                searchable_text = ' '.join(searchable_parts).lower()
                
                # Ensure ALL terms are found in the searchable text
                matched = all(
                    term.lower() in searchable_text 
                    for term in terms
                )
                
                if matched:
                    matches.add((payload.get('title', ''), json.dumps(payload)))

            return [json.loads(match[1]) for match in matches]

        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    def create_system_prompt(self) -> str:
        """Generate a comprehensive system prompt about the Sigma rules."""
        return f"""You have access to a comprehensive Sigma Rules Detection Database.

{self.rule_summary}

Capabilities:
- Search across 100+ security detection rules
- Coverage includes network, endpoint, and system-level threats
- Mapped to MITRE ATT&CK techniques
- Supports various search methods

Search Methods:
1. Direct keyword search
2. Phrase search with quotes
3. Contextual queries about security techniques

Recommended Query Formats:
- "What Sigma rules exist for network attacks?"
- search_qdrant:"windows authentication"
- "Explain detection methods for lateral movement"

When answering security-related questions, always consider searching and referencing the Sigma rules database to provide comprehensive, rule-backed insights."""

    def create_llm_prompt(self, query: str, context: str) -> str:
        """Create an enhanced prompt with multiple context sources."""
        # Semantic search for additional context
        semantic_rules = self._semantic_search(query)
        
        # Combine contexts
        expanded_context = context
        if semantic_rules:
            expanded_context += "\n\nSemantics-based Relevant Rules:\n"
            expanded_context += self.get_context_from_rules(semantic_rules)
        
        return f"""{self.create_system_prompt()}

Existing Context:
{expanded_context}

Query: {query}

Please provide a comprehensive response, leveraging the Sigma rules context. Refer to specific rules when applicable."""

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

    # Rest of the methods remain similar to previous implementation...
    # (includes pipe(), run(), and other utility methods)
