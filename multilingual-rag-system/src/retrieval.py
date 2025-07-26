from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def load_vector_store(index_path="faiss_index.bin", chunks_path="chunks.pkl"):
    """
    Load FAISS index and chunks from disk.
    """
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        embeddings, chunks = pickle.load(f)
    return index, embeddings, chunks

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    """
    Retrieve top-k relevant chunks for a given query.
    """
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [chunks[idx] for idx in indices[0]], distances[0]

if __name__ == "__main__":
    index, embeddings, chunks = load_vector_store()
    query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    relevant_chunks, distances = retrieve_relevant_chunks(query, index, chunks)
    for i, (chunk, dist) in enumerate(zip(relevant_chunks, distances)):
        print(f"Chunk {i+1} (Distance: {dist:.4f}): {chunk}")