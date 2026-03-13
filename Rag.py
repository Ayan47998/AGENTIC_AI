import os
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Set up paths
DATA_DIR = os.getenv("DATA_DIR")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR")

if not DATA_DIR:
    raise ValueError("DATA_DIR environment variable is not set.")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist.")

if not VECTORSTORE_DIR:
    raise ValueError("VECTORSTORE_DIR environment variable is not set.")
if not os.path.exists(VECTORSTORE_DIR):
    os.makedirs(VECTORSTORE_DIR) 

# Step 1: Load documents
def load_documents():
    documents = []
    files = os.listdir(DATA_DIR)
    if not files:
        raise FileNotFoundError(f"No files found in data directory '{DATA_DIR}'.")
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".docx"):
                loader = UnstructuredFileLoader(filepath)
            else:
                continue
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"⚠️ Error loading '{filename}': {e}")
    return documents

# Step 2: Split into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# Step 3: Set up embeddings
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create Chroma vectorstore
def create_vectorstore(splits, embeddings):
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    vectordb.persist()
    return vectordb

# Step 5: Main setup function
def build_vectorstore():
    print("Loading documents...")
    docs = load_documents()

    print(f"Loaded {len(docs)} documents. Splitting...")
    splits = split_documents(docs)

    print(f"Total {len(splits)} chunks. Embedding...")
    embeddings = get_embeddings_model()

    print("Creating Chroma vectorstore...")
    vectordb = create_vectorstore(splits, embeddings)

    print("✅ Vectorstore created and saved.")
    return vectordb


if __name__ == "__main__":
    build_vectorstore()

