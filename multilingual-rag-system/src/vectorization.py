from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def vectorize_chunks(chunks):
    """
    Convert text chunks to embeddings and store in FAISS index.
    """
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, chunks

def save_vector_store(index, embeddings, chunks, index_path="faiss_index.bin", chunks_path="chunks.pkl"):
    """
    Save FAISS index and chunks to disk.
    """
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump((embeddings, chunks), f)

if __name__ == "__main__":
    with open("chunks.txt", "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.read().split("\n\n") if line.strip()]
    index, embeddings, chunks = vectorize_chunks(chunks)
    save_vector_store(index, embeddings, chunks)