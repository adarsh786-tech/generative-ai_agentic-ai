import os
from pathlib import Path
from dotenv import load_dotenv
# langchain imports below
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_file_path = Path(__file__).parent / "node_handbook.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load() # gives pages

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents=docs)

embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


vector_store = QdrantVectorStore.from_documents(
    documents=[],
    collection_name="learning_langchain",
    embedding=embedder,
    url=os.getenv("QDRANT_URL"),         # ✅ Let langchain-qdrant create the client
    api_key=os.getenv("QDRANT_API_KEY")  # ✅ Same here
)

vector_store.add_documents(split_docs)
print("Injection Done!!")

retriver = QdrantVectorStore.from_existing_collection(
    collection_name="learning_langchain",
    embedding=embedder,
    url=os.getenv("QDRANT_URL"),         # ✅ Let langchain-qdrant create the client
    api_key=os.getenv("QDRANT_API_KEY")  # ✅ Same here
)

relevant_chunks = retriver.similarity_search(
    query="What is http module in JS?"
)

SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who responds based on the available context.

Context:
Author: {getattr(relevant_chunks[0].metadata, "author", "Unknown")}
Page Content:
{chr(10).join(chunk.page_content.replace("\\t", " ").replace("\t", " ") for chunk in relevant_chunks)}
"""

llmModel = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    ("system", SYSTEM_PROMPT),("human", "What is http module in JS?")   
]
llm_response = llmModel.invoke(messages)
print("Message Content: ", llm_response.content)