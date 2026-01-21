import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

def load_documents(k_base_path):
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader
    }
    documents = []
    for filename in os.listdir(k_base_path):
        file_path = os.path.join(k_base_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
        print(f"Loaded {len(documents)} documents from {k_base_path}")
    return documents

def split_documents(documents, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory='Ganko'):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectordb.add_documents(chunks)
    vectordb.persist()
    print(f"成功创建并持久化向量数据库到 '{persist_directory}' 目录。")
    return vectordb


def search_k_base(query,vectordb):
    retrieved_docs = vectordb.similarity_search(query, k=3)
    print(f"\n--- 为查询 '{query}' 检索到 {len(retrieved_docs)} 个相关文档 ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"【相关文档 {i+1}】\n来源: {doc.metadata.get('source', 'N/A')}\n内容: {doc.page_content[:200]}...\n")
    return retrieved_docs
