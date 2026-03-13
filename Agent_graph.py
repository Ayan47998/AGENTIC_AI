# agent_graph.py
import logging
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema.runnable import RunnableLambda
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from Retriver import get_vectorStore
from dotenv import load_dotenv

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_graph_debug.log")
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class AgentState(TypedDict):
    question: str
    result: str


# 🧠 Step 1: Load vectorstore retriever
retriever = get_vectorStore()
logger.info("Retriever loaded.")

# 🧠 Step 2: Initialize HuggingFaceEndpoint LLM
repo_id = os.getenv("HF_MODEL_URL", "google/flan-t5-large")  # <-- Changed model

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HF_API_KEY"),
    task="text2text-generation",  # Supported by flan-t5-large
    temperature=0,
    provider="auto"
)
logger.info(f"HuggingFaceEndpoint initialized with repo_id: {repo_id}")

# 🧠 Step 3: Build RAG chain with retrieval
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
)
logger.info("RAG chain initialized.")

# 🤖 Step 4: Define function node for LangGraph
@RunnableLambda
def query_node(state: AgentState) -> AgentState:
    question = state["question"]
    logger.info(f"Received question: {question}")
    try:
        docs = retriever.get_relevant_documents(question)
        logger.info(f"Retrieved {len(docs)} documents.")
        result = rag_chain.run(question)
        logger.info(f"RAG result: {result}")
    except Exception as e:
        result = f"Error: {e}"
        logger.error(f"Error in rag_chain.run: {e}", exc_info=True)
    return {"question": question, "result": result}

# 🔁 Step 5: LangGraph stateful execution
workflow = (
    StateGraph(AgentState)
    .add_node("query_node", query_node)
    .set_entry_point("query_node")
    .add_edge("query_node", END)
    .compile()
)

# 🚀 Step 6: Inference function
def run_query(question):
    logger.info(f"run_query called with: {question}")
    state = {"question": question, "result": ""}
    try:
        result = workflow.invoke(state)
        logger.info(f"Workflow result: {result}")
    except Exception as e:
        logger.error(f"Error in workflow.invoke: {e}", exc_info=True)
        result = {"result": f"Error: {e}"}
    return result["result"]

