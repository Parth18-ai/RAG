"""
Basic RAG pipeline
1) load a .txt document
2) chunk it
3) embed chunks with sentence-transformers
4) retrieve top-k chunks for a user query
5) pass them to an LLM (via Ollama) to write the answer
"""

import re
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# ---------- Settings you can change ----------
DOC_PATH = "faq.txt"          # the document to search
LLM_MODEL = "llama3.2:latest"          # run `ollama list` and use a model name you have
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 3                # sentences per chunk
OVERLAP = 1                   # sentences repeated between neighbouring chunks
TOP_K = 3                     # how many chunks to retrieve



def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    text = text.replace("\n", " ")
    # split into sentences at . ! ?
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break
    return chunks


def embed_chunks(chunks, model):
    # normalize_embeddings=True makes every vector length 1,
    # so a simple dot product equals cosine similarity
    return model.encode(chunks, normalize_embeddings=True)



def retrieve(query, chunks, chunk_vectors, model, k=TOP_K):
    query_vector = model.encode([query], normalize_embeddings=True)[0]
    scores = chunk_vectors @ query_vector           # similarity of query to every chunk
    top_indices = np.argsort(scores)[::-1][:k]      # highest scores first
    return [(chunks[i], float(scores[i])) for i in top_indices]



def ask_llm(query, retrieved):
    context = "\n\n".join(chunk for chunk, _ in retrieved)

    prompt = f"""Use ONLY the information below to answer the question.
If the answer is not in the information, say "I don't know".

Information:
{context}

Question: {query}
Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]



def main():
    print("Loading document and building the index...")
    text = load_document(DOC_PATH)
    chunks = chunk_text(text)

    embed_model = SentenceTransformer(EMBED_MODEL)
    chunk_vectors = embed_chunks(chunks, embed_model)
    print(f"Done: {len(chunks)} chunks ready.\n")

    while True:
        query = input("Ask a question (or type quit): ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        retrieved = retrieve(query, chunks, chunk_vectors, embed_model)

        print("\n--- Retrieved chunks ---")
        for chunk, score in retrieved:
            print(f"[score {score:.2f}] {chunk}")

        answer = ask_llm(query, retrieved)
        print("\n--- Answer ---")
        print(answer)
        print()


if __name__ == "__main__":
    main()
