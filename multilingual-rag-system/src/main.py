from extract_text import extract_text_with_ocr
from chunking import chunk_text
from vectorization import vectorize_chunks, save_vector_store
from retrieval import retrieve_relevant_chunks
from generation import initialize_llm, generate_answer
from langchain.memory import ConversationBufferMemory

def main():
    # Extract text
    pdf_path = "/home/tahsin/Desktop/Projects/RAG_system_10ms/multilingual-rag-system/data/HSC26-Bangla1st-Paper.pdf"
    text = extract_text_with_ocr(pdf_path)
    
    # Chunk text
    chunks = chunk_text(text)
    
    # Vectorize and store
    index, embeddings, chunks = vectorize_chunks(chunks)
    save_vector_store(index, embeddings, chunks)
    
    # Initialize LLM and memory
    tokenizer, llm = initialize_llm()
    memory = ConversationBufferMemory()
    
    # Test queries
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    
    for query in test_queries:
        relevant_chunks, _ = retrieve_relevant_chunks(query, index, chunks)
        answer = generate_answer(query, relevant_chunks, memory, tokenizer, llm)
        print(f"Query: {query}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()