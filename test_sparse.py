from sentence_transformers import SparseEncoder


print("Loading sparse model...")

model = SparseEncoder(
    "ibm-granite/granite-embedding-30m-sparse"
)

print("Model loaded.")


documents = [
    "The bidder must have a minimum annual turnover of fifty lakh rupees.",
    "The contractor must have five years of technical experience.",
    "The tender requires submission of an earnest money deposit."
]


print("Encoding documents...")

document_embeddings = model.encode_document(
    documents,
    max_active_dims=192
)


query = [
    "What is the minimum turnover requirement?"
]


print("Encoding query...")

query_embedding = model.encode_query(
    query,
    max_active_dims=50
)


scores = model.similarity(
    query_embedding,
    document_embeddings
)


print("\nSimilarity scores:")
print(scores)

print("\nBest document:")

best_index = scores[0].argmax().item()

print(
    documents[best_index]
)