from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Qdrant Setup
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "sigma_rules"

# Llama Setup
llm_predictor = LLMPredictor(llm_name="llama3.2:latest")
prompt_helper = PromptHelper(max_input_size=4096, context_window=2048)
index = GPTSimpleVectorIndex.from_existing_collection(
    collection_name=collection_name,
    qdrant_client=qdrant_client,
    llm_predictor=llm_predictor,
    prompt_helper=prompt_helper
)

# Query Function
def query_sigma(query_text):
    search_filter = Filter(
        must=[FieldCondition(key="tags", match=MatchValue(value="sigma"))]
    )
    response = qdrant_client.search(
        collection_name=collection_name,
        query_vector=llm_predictor.embed_query(query_text),
        limit=5,
        query_filter=search_filter
    )
    if response:
        for res in response:
            print(f"Match: {res.payload['rule_name']}")
        return [res.payload["content"] for res in response]
    return ["No matching Sigma rules found."]

# Example Usage
user_query = "Detect lateral movement with SMB"
results = query_sigma(user_query)
for result in results:
    print(result)
