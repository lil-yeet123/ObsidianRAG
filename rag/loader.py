from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents(path: str):
    loader = DirectoryLoader(
        path,
        glob="**/*.md",
        loader_cls=lambda p: TextLoader(p, encoding="utf-8"),
        show_progress=True,
    )
    return loader.load()

