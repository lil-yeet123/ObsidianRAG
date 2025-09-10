import os
import sys
import threading
import time

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.callbacks import BaseCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag.loader import load_documents
from rag.qa import build_qa_chain
from rag.vectorstore import load_vectorstore, build_vectorstore

DB_PATH = "db/faiss_index"


VAULT_PATH = "G:/notes/Obsidian"




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
    documents = load_documents(VAULT_PATH)
    documents = [doc for doc in documents if "Templates/" not in doc.metadata["source"]]
    print(f"[+] Geladene Dokumente: {len(documents)}")

    db = build_vectorstore(documents, persist_directory=DB_PATH)
    return db

def chat():
    if not os.path.exists(DB_PATH) or not os.path.exists(os.path.join(DB_PATH, "faiss_index.pkl")):
        db = init_db()
    else:
        db = load_vectorstore(persist_directory=DB_PATH)

    qa = build_qa_chain(db, model_name="gemma2:9b")
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
