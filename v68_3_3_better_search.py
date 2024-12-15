"""
title: Qdrant Sigma Rules Pipeline - Enhanced
author: open-webui
date: 2024-12-14
version: 1.1
license: MIT
description: Improved pipeline for searching and analyzing Sigma rules using Qdrant and LLM
requirements: qdrant-client, requests
"""

from typing import List, Dict, Any, Generator
import logging
import json
import re
import os
import requests
from qdrant_client import QdrantClient
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
                "SEARCH_PREFIX": os.getenv("SEARCH_PREFIX", "search_qdrant:")
            }
        )

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )

    async def on_startup(self):
        """Verify connections on startup."""
        try:
            collection_info = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
            print(f"Connected to Qdrant collection: {collection_info}")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise

    async def on_shutdown(self):
        """Clean up resources."""
        pass

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms, with special handling for custom search prefix and rule references."""
        # Check for custom search prefix
        search_prefix = self.valves.SEARCH_PREFIX
        if query.startswith(search_prefix):
            # Remove the prefix and strip whitespace
            search_query = query[len(search_prefix):].strip()
            
            # First try exact phrase matching
            phrases = re.findall(r'"([^"]*)"', search_query)
            if phrases:
                return phrases
            
            # If no phrases, split the remaining query
            return [term.strip() for term in search_query.split() if term.strip()]
        
        # Existing search term extraction for regular queries
        phrases = re.findall(r'"([^"]*)"', query)
        if not phrases:
            # Check for rule title references
            rule_title_match = re.search(r'Rule\s*"([^"]+)"', query)
            if rule_title_match:
                return [rule_title_match.group(1)]
            
            # Look for terms after "about" or "related to"
            about_match = re.search(r'about\s+"?([^"]+)"?', query.lower())
            related_match = re.search(r'related to\s+"?([^"]+)"?', query.lower())
            if about_match:
                return [about_match.group(1).strip()]
            if related_match:
                return [related_match.group(1).strip()]
            
            # Standard term extraction
            terms = [term.strip() for term in query.split() if term.strip()]
            return terms
        return phrases

    def advanced_search_qdrant(self, terms: List[str], advanced_search: bool = False) -> List[Dict]:
        """Enhanced search for Sigma rules with more sophisticated matching."""
        try:
            result = self.qdrant.scroll(
                collection_name=self.valves.QDRANT_COLLECTION,
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            matches = set()
            if result and result[0]:
                for point in result[0]:
                    payload = point.payload
                    searchable_parts = []
                    
                    # Comprehensive field extraction
                    search_fields = {
                        'simple': ['title', 'id', 'status', 'author', 'date', 'modified', 'level', 'filename'],
                        'complex': ['description', 'references', 'tags', 'falsepositives'],
                        'nested': ['logsource', 'detection']
                    }
                    
                    # Simple fields
                    for field in search_fields['simple']:
                        if payload.get(field):
                            searchable_parts.append(str(payload[field]))
                    
                    # Complex fields
                    for field in search_fields['complex']:
                        field_value = payload.get(field)
                        if field_value:
                            if isinstance(field_value, list):
                                searchable_parts.extend(str(item) for item in field_value)
                            else:
                                searchable_parts.append(str(field_value))
                    
                    # Nested fields
                    if payload.get('logsource'):
                        searchable_parts.extend(str(v) for v in payload['logsource'].values())
                    
                    if payload.get('detection'):
                        detection_str = json.dumps(payload['detection'])
                        searchable_parts.append(detection_str)
                    
                    # MITRE ATT&CK tag handling
                    if payload.get('tags'):
                        attack_tags = [tag for tag in payload['tags'] if tag.startswith('attack.')]
                        if attack_tags:
                            searchable_parts.extend(attack_tags)
                    
                    # Combine searchable parts
                    searchable_text = ' '.join(searchable_parts).lower()
                    
                    # Enhanced matching with special handling for full title matches
                    title_match = False
                    partial_match_score = 0
                    
                    for term in terms:
                        # Exact title match gets top priority
                        if term.lower() == payload.get('title', '').lower():
                            title_match = True
                        
                        # Partial matching logic
                        if term.lower() in searchable_text or \
                           any(term.lower() in word.lower() for word in searchable_text.split()):
                            partial_match_score += 1
                    
                    # Decide whether to add the match
                    if title_match or (partial_match_score > 0 and advanced_search):
                        match_details = (
                            payload.get('title', ''), 
                            json.dumps(payload),
                            3 if title_match else partial_match_score
                        )
                        matches.add(match_details)

            # Sort matches by score, with title matches first
            sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
            return [json.loads(match[1]) for match in sorted_matches]

        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and return results."""
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            # Check for custom search prefix
            search_prefix = self.valves.SEARCH_PREFIX
            is_custom_search = query.startswith(search_prefix)
            
            # Extract potential search terms
            search_terms = self.extract_search_terms(query)
            
            # Determine search type
            advanced_search = is_custom_search
            
            # Always search Qdrant first to get potential context
            matches = self.advanced_search_qdrant(search_terms, advanced_search) if search_terms else []
            
            # If it's a direct search request, show the rules
            if is_custom_search or self.looks_like_search(query):
                if matches:
                    yield f"Found {len(matches)} matching Sigma rules:\n\n"
                    for idx, rule in enumerate(matches, 1):
                        # Rule number and title outside of code block
                        yield f"Rule {idx}: {rule.get('title', 'Untitled')}\n"
                        # Rule content in code block
                        yield "```yaml\n"
                        yield self.format_rule(rule)
                        yield "\n```\n\n"
                    return
                else:
                    yield f"No Sigma rules found matching: {', '.join(search_terms)}\n"
                    return
            
            # If it's a question and context is enabled, include matching rules
            if self.looks_like_question(query) and matches and self.valves.ENABLE_CONTEXT:
                context = self.get_context_from_rules(matches)
                llm_prompt = self.create_llm_prompt(query, context)
            else:
                llm_prompt = query

            # Get LLM response
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
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            yield f"Error: {str(e)}"

    def looks_like_search(self, query: str) -> bool:
        """Check if query is a search request."""
        if '"' in query:
            return True
        search_words = ['search', 'find', 'show', 'list', 'get']
        query_words = query.lower().split()
        if any(word in query_words for word in search_words):
            return True
        return len(query_words) <= 3

    def looks_like_question(self, query: str) -> bool:
        """Check if query is a question about rules."""
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain']
        contains_about = 'about' in query.lower()
        starts_with_question = any(query.lower().startswith(word) for word in question_words)
        return starts_with_question or contains_about

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
            if field == 'description':
                if value:
                    yaml_output.append(f"description: |")
                    for line in str(value).split('\n'):
                        yaml_output.append(f"    {line.strip()}")
                else:
                    yaml_output.append("description: |")
                    yaml_output.append("    No description provided")
                    
            elif field in ['logsource', 'detection']:
                yaml_output.append(f"{field}:")
                if isinstance(value, dict):
                    dict_lines = json.dumps(value, indent=4).split('\n')
                    for line in dict_lines[1:-1]:
                        yaml_output.append(f"    {line.strip()}")
                else:
                    yaml_output.append("    {}")
                    
            elif isinstance(value, list):
                yaml_output.append(f"{field}:")
                if value:
                    for item in value:
                        yaml_output.append(f"    - {item}")
                else:
                    yaml_output.append("    - none")
                    
            elif isinstance(value, dict):
                yaml_output.append(f"{field}:")
                if value:
                    dict_lines = json.dumps(value, indent=4).split('\n')
                    for line in dict_lines[1:-1]:
                        yaml_output.append(f"    {line.strip()}")
                else:
                    yaml_output.append("    {}")
                    
            else:
                yaml_output.append(f"{field}: {value if value is not None else 'none'}")
                
            if field in ['description', 'detection', 'logsource']:
                yaml_output.append("")
                
        return '\n'.join(yaml_output)

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

    def create_llm_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM that includes context."""
        return f"""Here are some relevant Sigma detection rules for context:

{context}

Based on these rules, please answer this question:
{query}

Please be specific and refer to the rules when applicable."""

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run pipeline and return results."""
        results = list(self.pipe(prompt=prompt, **kwargs))
        if not results:
            return []
        return [{"text": "".join(results)}]
