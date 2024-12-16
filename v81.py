"""
title: Enhanced Qdrant Sigma Rules Pipeline with Context
author: open-webui
date: 2024-12-15
version: 1.2
license: MIT
description: Improved pipeline for searching and analyzing Sigma rules using Qdrant and LLM with conversation context
requirements: qdrant-client, requests, pydantic
"""

from typing import List, Dict, Any, Generator, Tuple
import logging
import json
import re
import os
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from qdrant_client import QdrantClient
from pydantic import BaseModel

@dataclass
class ConversationContext:
    last_rules: List[Dict] = field(default_factory=list)
    last_query: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_valid(self, timeout_minutes: int = 5) -> bool:
        """Check if context is still valid within timeout window."""
        return datetime.now() - self.timestamp < timedelta(minutes=timeout_minutes)

class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str
        QDRANT_PORT: int
        QDRANT_COLLECTION: str
        LLM_MODEL_NAME: str
        LLM_BASE_URL: str
        ENABLE_CONTEXT: bool
        CONTEXT_TIMEOUT: int
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
                "ENABLE_CONTEXT": True,  # Always enable context
                "CONTEXT_TIMEOUT": int(os.getenv("CONTEXT_TIMEOUT", "5")),
                "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
                "SEARCH_PREFIX": os.getenv("SEARCH_PREFIX", "search_qdrant:")
            }
        )

        # Initialize logging
        logging.basicConfig(level=getattr(logging, self.valves.LOG_LEVEL))
        self.logger = logging.getLogger(__name__)

        # Initialize context and Qdrant client
        self.context = ConversationContext()
        self.qdrant = QdrantClient(
            host=self.valves.QDRANT_HOST,
            port=self.valves.QDRANT_PORT
        )

    async def on_startup(self):
        """Verify connections on startup."""
        try:
            collection_info = self.qdrant.get_collection(self.valves.QDRANT_COLLECTION)
            self.logger.info(f"Connected to Qdrant collection: {collection_info}")
        except Exception as e:
            self.logger.error(f"Error connecting to Qdrant: {e}")
            raise

    async def on_shutdown(self):
        """Clean up resources."""
        pass

    def extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms with special handling for custom search prefix."""
        search_prefix = self.valves.SEARCH_PREFIX
        if query.startswith(search_prefix):
            search_query = query[len(search_prefix):].strip()
            phrases = re.findall(r'"([^"]*)"', search_query)
            if phrases:
                return phrases
            return [term.strip() for term in search_query.split() if term.strip()]
        
        phrases = re.findall(r'"([^"]*)"', query)
        if not phrases:
            about_match = re.search(r'about\s+(\w+)', query.lower())
            related_match = re.search(r'related to\s+(\w+)', query.lower())
            if about_match:
                return [about_match.group(1)]
            if related_match:
                return [related_match.group(1)]
            return [term.strip() for term in query.split() if term.strip()]
        return phrases

    def extract_rule_reference(self, query: str) -> Tuple[bool, str, int]:
        """Extract rule references from queries about previous rules."""
        # Check for direct rule number references
        rule_num_match = re.search(r'rule\s+(\d+)', query.lower())
        if rule_num_match:
            return True, "rule_number", int(rule_num_match.group(1))
            
        # Check for "the rule you just showed" type queries
        last_rule_patterns = [
            r'(that|the|this) rule',
            r'(last|previous) rule',
            r'rule you (just )?(showed|mentioned|displayed)',
            r'the one you (just )?(showed|mentioned|displayed)',
            r'what you (just )?(showed|mentioned|displayed)',
        ]
        
        for pattern in last_rule_patterns:
            if re.search(pattern, query.lower()):
                return True, "last_rule", 1
                
        return False, "", 0

    def get_referenced_rules(self, query: str) -> List[Dict]:
        """Get rules based on conversation context and query."""
        if not self.context.is_valid(self.valves.CONTEXT_TIMEOUT):
            return []
            
        is_reference, ref_type, rule_num = self.extract_rule_reference(query)
        
        if not is_reference or not self.context.last_rules:
            return []
            
        if ref_type == "rule_number" and 1 <= rule_num <= len(self.context.last_rules):
            return [self.context.last_rules[rule_num - 1]]
        elif ref_type == "last_rule":
            return [self.context.last_rules[0]]
            
        return []

    def advanced_search_qdrant(self, terms: List[str], advanced_search: bool = False) -> List[Dict]:
        """Enhanced search for Sigma rules with sophisticated matching."""
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
                    
                    # Extract searchable content from all fields
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
                    
                    if payload.get('logsource'):
                        searchable_parts.extend(str(v) for v in payload['logsource'].values())
                    
                    if payload.get('detection'):
                        detection_str = json.dumps(payload['detection'])
                        searchable_parts.append(detection_str)
                    
                    # Special handling for MITRE ATT&CK tags
                    if payload.get('tags'):
                        attack_tags = [tag for tag in payload['tags'] if tag.startswith('attack.')]
                        if attack_tags:
                            searchable_parts.extend(attack_tags)
                    
                    searchable_text = ' '.join(searchable_parts).lower()
                    
                    if advanced_search:
                        match_score = sum(
                            1 for term in terms 
                            if term.lower() in searchable_text or 
                            any(term.lower() in word.lower() for word in searchable_text.split())
                        )
                        
                        if match_score > 0:
                            matches.add((
                                payload.get('title', ''), 
                                json.dumps(payload),
                                match_score
                            ))
                    else:
                        for term in terms:
                            if term.lower() in searchable_text:
                                matches.add((payload.get('title', ''), json.dumps(payload)))
                                break

            if advanced_search:
                sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
                return [json.loads(match[1]) for match in sorted_matches]
            
            return [json.loads(match[1]) for match in matches]

        except Exception as e:
            self.logger.error(f"Qdrant search error: {e}")
            return []

    def looks_like_search(self, query: str) -> bool:
        """Check if query is a search request."""
        if '"' in query:
            return True
        search_words = ['search', 'find', 'show', 'list', 'get']
        query_words = query.lower().split()
        return any(word in query_words for word in search_words) or len(query_words) <= 3

    def looks_like_question(self, query: str) -> bool:
        """Check if query is a question about rules."""
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'tell']
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
        if not rules:
            return ""
            
        context = []
        for idx, rule in enumerate(rules, 1):
            context.append(f"Rule {idx}:")
            context.append(f"Title: {rule.get('title', 'Untitled')}")
            if rule.get('description'):
                context.append(f"Description: {rule['description']}")
            if rule.get('detection'):
                context.append("Detection:")
                context.append(json.dumps(rule['detection'], indent=2))
            if rule.get('tags'):
                context.append(f"Tags: {', '.join(rule['tags'])}")
            context.append("---")
        return '\n'.join(context)

    def create_llm_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM that includes context."""
        if not context:
            return query
            
        return f"""Here are some relevant Sigma detection rules for context:

{context}

Based on these rules, please answer this question:
{query}

Please be specific and refer to the rules when applicable."""

    def get_llm_response(self, prompt: str) -> Generator[str, None, None]:
        """Get streaming response from LLM."""
        try:
            response = requests.post(
                url=f"{self.valves.LLM_BASE_URL}/api/generate",
                json={"model": self.valves.LLM_MODEL_NAME, "prompt": prompt},
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
            self.logger.error(f"Error getting LLM response: {e}")
            yield f"Error: Unable to get response from LLM - {str(e)}"

    def pipe(self, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Process input and return results with context awareness."""
        query = prompt or kwargs.get('user_message', '')
        if not query:
            return

        try:
            # First check if the query references previous rules
            referenced_rules = self.get_referenced_rules(query)
            if referenced_rules:
                self.logger.info(f"Found referenced rules: {len(referenced_rules)}")
                context = self.get_context_from_rules(referenced_rules)
                llm_prompt = self.create_llm_prompt(query, context)
                yield from self.get_llm_response(llm_prompt)
                return

            # If no specific reference, proceed with normal search
            search_terms = self.extract_search_terms(query)
            matches = self.advanced_search_qdrant(search_terms) if search_terms else []
            
            # Update context with new results
            if matches:
                self.context.last_rules = matches
                self.context.last_query = query
                self.context.timestamp = datetime.now()
                self.logger.info(f"Updated context with {len(matches)} new rules")

            # Handle direct search requests
            if self.looks_like_search(query):
                if matches:
                    yield f"Found {len(matches)} matching Sigma rules:\n\n"
                    for idx, rule in enumerate(matches, 1):
                        yield f"Rule {idx}: {rule.get('title', 'Untitled')}\n"
                        yield "```yaml\n"
                        yield self.format_rule(rule)
                        yield "\n```\n\n"
                else:
                    yield f"No Sigma rules found matching: {', '.join(search_terms)}\n"
                return

            # For questions, include context if available
            if self.looks_like_question(query):
                context = self.get_context_from_rules(matches if matches else self.context.last_rules)
                llm_prompt = self.create_llm_prompt(query, context)
                yield from self.get_llm_response(llm_prompt)
            else:
                yield from self.get_llm_response(query)

        except Exception as e:
            self.logger.error(f"Error in pipe: {e}")
            yield f"Error: {str(e)}"

    def run(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """Run pipeline and return results."""
        try:
            results = list(self.pipe(prompt=prompt, **kwargs))
            if not results:
                return []
            return [{"text": "".join(results)}]
        except Exception as e:
            self.logger.error(f"Error in run: {e}")
            return [{"text": f"Error: {str(e)}"}]

if __name__ == "__main__":
    # Example usage
    pipeline = Pipeline()
    
    # Example queries to test
    test_queries = [
        "search for suspicious processes",
        "tell me about rule 1",
        "what does this rule detect?",
        "search_qdrant:powershell",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = pipeline.run(query)
        print("Response:", results[0]["text"] if results else "No response")
