from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

def build_qa_chain(db, model_name="mistral"):
    llm = ChatOllama(model=model_name)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        return_source_documents=True
    )
    return qa
