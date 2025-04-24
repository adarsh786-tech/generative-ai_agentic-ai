# PDF Chatbot with LangChain and Qdrant

This project demonstrates how to build a chatbot that processes a PDF document, stores its embeddings in a Qdrant vector store, and retrieves relevant information using LangChain and Google Generative AI.

## Features

- **PDF Parsing**: Extracts text from a PDF file using `PyPDFLoader`.
- **Text Splitting**: Splits the extracted text into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Embeddings**: Generates embeddings for the text chunks using `GoogleGenerativeAIEmbeddings`.
- **Vector Store**: Stores and retrieves embeddings using `QdrantVectorStore`.
- **Generative AI**: Uses `ChatGoogleGenerativeAI` to generate responses based on retrieved context.

## Prerequisites

1. Python 3.8 or higher.
2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
