import chromadb
from sentence_transformers import SentenceTransformer

# Create embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="tourism_knowledge"
)


def load_documents():

    with open("data/tourism.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # Split text into chunks
    chunks = [
        text[i:i + 500]
        for i in range(0, len(text), 500)
    ]

    # Add documents only if collection is empty
    existing = collection.get()

    if len(existing["ids"]) == 0:

        embeddings = embedding_model.encode(
            chunks
        ).tolist()

        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

    return len(chunks)


def retrieve_context(query, n_results=3):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    documents = results["documents"][0]

    return "\n\n".join(documents)