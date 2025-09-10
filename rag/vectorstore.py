from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
import os

# Build Vectorstore mit FAISS
def build_vectorstore(documents, persist_directory="db/faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings-Modell
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS Index bauen
    db = FAISS.from_documents(docs, embeddings)

    # Optional: FAISS Index abspeichern
    os.makedirs(persist_directory, exist_ok=True)
    with open(os.path.join(persist_directory, "faiss_index.pkl"), "wb") as f:
        pickle.dump(db, f)

    return db

# FAISS Index laden
def load_vectorstore(persist_directory="db/faiss_index"):
    with open(os.path.join(persist_directory, "faiss_index.pkl"), "rb") as f:
        db = pickle.load(f)
    return db
