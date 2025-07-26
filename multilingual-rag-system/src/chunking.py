from langchain.text_splitter import RecursiveCharacterTextSplitter
from indicnlp.tokenize import indic_tokenize

def chunk_text(text, chunk_size=200, chunk_overlap=20):
    """
    Split text into chunks for semantic retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(indic_tokenize.trivial_tokenize(x, lang='bn'))
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    with open("extracted_text.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    with open("chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")