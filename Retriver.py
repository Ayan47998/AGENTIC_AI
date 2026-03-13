import logging
import pdb
from langchain.vectorstores import Chroma
from Rag import get_embeddings_model

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("retriever_debug.log")
    ]
)
logger = logging.getLogger(__name__)

def get_vectorStore():
    logger.info("Initializing Chroma vectorstore...")
    try:
        db = Chroma(persist_directory="vectorstore", embedding_function=get_embeddings_model())
        retriever = db.as_retriever(search_kwargs={"k": 3})
        logger.info("Chroma vectorstore and retriever initialized successfully.")
        # pdb.set_trace()  # Uncomment to start debugger here
    except Exception as e:
        logger.error(f"Error initializing vectorstore: {e}", exc_info=True)
        raise
    return retriever