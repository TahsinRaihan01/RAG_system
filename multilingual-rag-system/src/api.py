from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import load_vector_store, retrieve_relevant_chunks
from generation import initialize_llm, generate_answer
from langchain.memory import ConversationBufferMemory

app = FastAPI()
memory = ConversationBufferMemory()
tokenizer, llm = initialize_llm()
index, embeddings, chunks = load_vector_store()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    API endpoint to handle user queries and return generated answers.
    """
    relevant_chunks, distances = retrieve_relevant_chunks(request.query, index, chunks)
    answer = generate_answer(request.query, relevant_chunks, memory, tokenizer, llm)
    return {
        "query": request.query,
        "answer": answer,
        "relevant_chunks": relevant_chunks,
        "distances": distances.tolist()
    }