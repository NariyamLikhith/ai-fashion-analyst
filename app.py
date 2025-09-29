import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# --- NEW IMPORTS FOR MEMORY ---
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# --- SETUP ---

st.set_page_config(page_title="AI Fashion Trend Analyst 👠", layout="wide")

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
except KeyError:
    st.error("Groq or Cohere API key not found! Please add them to your Streamlit secrets.")
    st.stop()

# --- DATA INGESTION AND VECTORIZATION ---

@st.cache_resource
def load_and_vectorize_data():
    with st.spinner("Analyzing Fashion Reports... This may take a few minutes on first run."):
        
        loader = DirectoryLoader("fashion_reports", glob="**/*.txt", show_progress=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = Qdrant.from_documents(
            final_documents,
            embeddings,
            location=":memory:",
            collection_name="fashion_trends",
        )
    return vectorstore

# --- AI AND RETRIEVAL LOGIC ---

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
vectorstore = load_and_vectorize_data()
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-english-v3.0", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor= compressor, base_retriever=base_retriever
)


# --- NEW: PROMPT FOR HISTORY-AWARE RETRIEVER ---
# This prompt helps the AI rephrase the user's question to be a standalone question
# based on the chat history.
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, compression_retriever, contextualize_q_prompt
)


# --- NEW: PROMPT FOR THE FINAL ANSWER ---
# This is the prompt that will be used to generate the final answer, using the retrieved documents.
qa_system_prompt = (
    "You are an expert fashion industry analyst. Use the following retrieved context "
    "to answer the user's question. Your answer should be detailed, insightful, "
    "and synthesized from the provided information."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# --- UPDATED CHAIN CREATION WITH MEMORY ---
# We create the main chain that combines the history-aware retriever and the answering prompt.
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- STREAMLIT UI ---

st.title("AI Fashion Trend Analyst 👠")
st.markdown("Ask me about the latest sustainability trends in the fashion industry!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # --- NEW: Display sources if they exist in the message ---
        if "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)


if user_prompt := st.chat_input("What are the key sustainability trends in fashion?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("Analyzing..."):
        
        # --- UPDATED: Convert session state to chat history format ---
        chat_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # --- UPDATED: Invoke chain with history ---
        result = rag_chain.invoke({"input": user_prompt, "chat_history": chat_history})
        ai_response = result["answer"]

        # --- NEW: Get and format the sources ---
        source_documents = result["context"]
        source_names = set()
        if source_documents:
            for doc in source_documents:
                # Extracts the filename from the source path
                source_names.add(doc.metadata['source'].split('/')[-1].split('\\')[-1])
        
        # Add the AI response and sources to the session state
        assistant_message = {"role": "assistant", "content": ai_response}
        if source_names:
            assistant_message["sources"] = list(source_names)
        
        st.session_state.messages.append(assistant_message)
        
        # Display the AI response and sources
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            if source_names:
                with st.expander("Sources"):
                    for source in source_names:
                        st.write(source)