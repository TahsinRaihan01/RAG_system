from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

def evaluate_rag(query, answer, chunks, expected_answer):
    """
    Evaluate RAG system for groundedness and relevance.
    """
    # Groundedness: Check if answer is in retrieved chunks
    grounded = any(answer.lower() in chunk.lower() for chunk in chunks)
    
    # Relevance: Cosine similarity between query and chunks
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = model.encode([query])
    chunk_embeddings = model.encode(chunks)
    relevance_score = np.mean(cosine_similarity(query_embedding, chunk_embeddings))
    
    # Accuracy: Check if answer matches expected answer
    accuracy = answer.lower() == expected_answer.lower()
    
    return {
        "grounded": grounded,
        "relevance_score": float(relevance_score),
        "accuracy": accuracy
    }

if __name__ == "__main__":
    from retrieval import load_vector_store, retrieve_relevant_chunks
    index, embeddings, chunks = load_vector_store()
    test_cases = [
        ("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "শুম্ভুনাথ"),
        ("কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "মামাকে"),
        ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "১৫ বছর")
    ]
    for query, expected in test_cases:
        relevant_chunks, _ = retrieve_relevant_chunks(query, index, chunks)
        # Simulate answer (replace with actual generation in practice)
        answer = expected  # Placeholder
        result = evaluate_rag(query, answer, relevant_chunks, expected)
        print(f"Query: {query}\nResult: {result}")