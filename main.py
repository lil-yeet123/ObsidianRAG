import os
import sys
import threading
import time

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.callbacks import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag.loader import load_documents
from rag.qa import build_qa_chain
from rag.vectorstore import load_vectorstore

DB_PATH = "chroma_db"

VAULT_PATH = "/home/matti/Obsidian"




def spinner_task(stop_event):
    spinner = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{spinner[idx % len(spinner)]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 20 + "\r")


def init_db():
    loader = DirectoryLoader(VAULT_PATH, glob="**/*.md")
    documents = [doc for doc in loader.load() if "Templates/" not in doc.metadata["source"]]
    print(f"[+] Geladene Dokumente: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)
    print(f"[+] Nach Split: {len(texts)} Chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    db.persist()
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

        print("\n", end=" ", flush=True)
        result = qa.invoke({"query": query})  # Tokens kommen live über StreamingHandler
        print("\n")  # sauberer Zeilenumbruch nach kompletter Antwort

        if "source_documents" in result:
            print("\n📂 Quellen:")
            for doc in result["source_documents"]:
                print("-", doc.metadata.get("source"))
        print("\n")

if __name__ == "__main__":
    chat()
