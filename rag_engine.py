import os
import uuid
from typing import Dict, List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ----------------------------------
# Config
# ----------------------------------
BASE_DIR = os.path.dirname(__file__)
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

_vectorstore = None
_embeddings = None
_llm = None


# ----------------------------------
# Vectorstore Loader
# ----------------------------------
def get_vectorstore():
    global _vectorstore, _embeddings

    if _vectorstore is None:

        if _embeddings is None:
            _embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )

        _vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=_embeddings,
            collection_name="rag_collection",
        )

    return _vectorstore


# ----------------------------------
# LLM Loader
# ----------------------------------
def get_llm():
    global _llm

    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

    return _llm


# ----------------------------------
# Index Document
# ----------------------------------
def index_document(filename: str, text: str) -> None:
    if not text.strip():
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(text)

    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "source": filename,
                "chunk_id": str(uuid.uuid4()),
            },
        )
        for chunk in chunks
    ]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)


# ----------------------------------
# Helper: Format Documents
# ----------------------------------
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ----------------------------------
# Query RAG (LCEL STYLE)
# ----------------------------------
def query_rag(question: str) -> Dict:
    vectorstore = get_vectorstore()
    llm = get_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    system_prompt = (
        "You are an intelligent RAG assistant similar to NotebookLM.\n"
        "Use ONLY the provided context to answer.\n"
        "If the answer is unknown, say you don't know.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # -----------------------------
    # NEW LCEL PIPELINE
    # -----------------------------
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    # Fetch sources separately (modern pattern)
    docs = retriever.invoke(question)

    sources = list({
        doc.metadata.get("source")
        for doc in docs
    })

    return {
        "answer": answer,
        "sources": sources,
    }