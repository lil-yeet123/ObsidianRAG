from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


def build_qa_chain(db, model_name="gemma2:9b"):
    llm = ChatOllama(
        model=model_name,
        callbacks=[StreamingHandler()],
    )

    # System-Anweisung
    system_template = """
    Du bist ein persönlicher Assistent mit Zugriff auf ein Obsidian-Vault.
    Deine Hauptaufgabe ist es, Fragen des Nutzers anhand seines Wissensarchivs zu beantworten.
    
    Regeln:
    
    - Nutze ausschließlich die bereitgestellten kontextuellen Informationen aus dem Obsidian-Vault.
    - Falls der Kontext unzureichend ist, sage das offen.
    - Formuliere klar, präzise und strukturiert.
    - Verweise auf Quellen, wenn möglich.
    - Rückfragen stellen, falls die Frage zu vage ist.
    - Ergänze Allgemeinwissen nur gekennzeichnet.
    - Nutze eine sachliche, freundliche Ausdrucksweise.
    - Antworte in der Sprache der Nutzerfrage (Deutsch/Englisch).
    - Antworte niemals in Markdown außer du wirst explicit darum gebeten.
    """

    system_message = SystemMessagePromptTemplate.from_template(system_template)

    human_template = """
    📂 Kontext aus den Dokumenten:
    {context}

    ❓ Frage:
    {question}
    """
    human_message = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": chat_prompt,
            "document_variable_name": "context"  # <-- hier!
        }
    )

    return qa
