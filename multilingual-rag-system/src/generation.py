from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

def initialize_llm():
    """
    Initialize the multilingual LLM and tokenizer.
    """
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")  # Use slow tokenizer
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    return tokenizer, llm

def generate_answer(query, chunks, memory, tokenizer, llm):
    """
    Generate an answer based on query, retrieved chunks, and chat history.
    """
    context = "\n".join(chunks)
    history = memory.load_memory_variables({})['history']
    prompt = f"History: {history}\nContext: {context}\nQuery: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = llm.generate(**inputs, max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    memory.save_context({"query": query}, {"answer": answer})
    return answer

if __name__ == "__main__":
    from retrieval import load_vector_store, retrieve_relevant_chunks
    index, embeddings, chunks = load_vector_store()
    query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    relevant_chunks, _ = retrieve_relevant_chunks(query, index, chunks)
    memory = ConversationBufferMemory()
    tokenizer, llm = initialize_llm()
    answer = generate_answer(query, relevant_chunks, memory, tokenizer, llm)
    print(f"Query: {query}\nAnswer: {answer}")