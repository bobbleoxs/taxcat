from haystack.document_stores.in_memory import InMemoryDocumentStore

store = InMemoryDocumentStore()
all_chunks = store.filter_documents()
print(f"Total chunks in store: {len(all_chunks)}")
# Search within chunks for "zero-rated"
zero_chunks = [chunk for chunk in all_chunks if "zero-rated" in chunk.content.lower()]
print(f"Chunks containing 'zero-rated': {len(zero_chunks)}")
for idx, chunk in enumerate(zero_chunks):
    print(f"\n--- Chunk #{idx+1} ---")
    print(chunk.content[:200].replace("\n", " "), "...")
