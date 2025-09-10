import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag.loader import load_documents
from rag.qa import build_qa_chain
from rag.vectorstore import load_vectorstore

DB_PATH = "db"

VAULT_PATH = "/home/matti/Obsidian"


def init_db():
    print("[*] Lade Dokumente...")
    documents = load_documents(VAULT_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("[*] Erstelle Vektorstore...")
    db = Chroma.from_documents(documents, embeddings)

    return db


def chat():
    if not os.path.exists(DB_PATH):
        db = init_db()
    else:
        db = load_vectorstore(DB_PATH)

    qa = build_qa_chain(db, model_name="mistral")

    print("Obsidian-RAG-Bot (quit mit 'exit')")
    while True:
        query = input("Frage: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa.invoke({"query": query})
        print("\n💡 Antwort:", result["result"])
        if "source_documents" in result:
            print("\n📂 Quellen:")
            for doc in result["source_documents"]:
                print("-", doc.metadata.get("source"))
        print("\n")


if __name__ == "__main__":
    chat()
