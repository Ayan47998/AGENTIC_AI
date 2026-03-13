import streamlit as st
import logging
import sys
from Agent_graph import run_query

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app_debug.log")
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="FinTrust Contract Assistant", layout="wide")
st.title("📄 FinTrust Financial Contract Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.info("Enter a question related to financial agreements or loan terms.")
    example_q = st.selectbox("Try examples", [
        "What are the repayment terms in the master loan agreement?",
        "Who are the parties involved in the investment agreement?",
        "What is the penalty clause in the promissory note?",
        "List key terms from the SME term sheet."
    ])

query = st.text_input("Ask a question about FinTrust contracts:", value=example_q)

if st.button("Submit") and query:
    logger.info(f"User submitted query: {query}")
    with st.spinner("Thinking..."):
        try:
            answer = run_query(query)
            logger.info(f"Answer generated: {answer}")
        except Exception as e:
            answer = f"Error: {e}"
            logger.error(f"Error during run_query: {e}", exc_info=True)
    st.session_state.chat_history.append((query, answer))

if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"{i}. {q}"):
            st.write(a)
            logger.debug(f"Chat history item {i}: Q={q}, A={a}")
