from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate


class StreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


def build_qa_chain(db, model_name="mistral"):
    llm = ChatOllama(model=model_name,
                     callbacks=[StreamingHandler()],)

    template = """
    Antworte immer auf Deutsch.
    Nutze die folgenden Dokumente, um die Frage zu beantworten:

    {context}

    Frage: {question}
    Antwort (Deutsch):
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa
