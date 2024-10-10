import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize GROQ API key
groq_api_key = os.getenv("GROQ_API_KEY")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details and use bullet points where appropriate. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide incorrect answers.
    
    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """

    model = ChatGroq(
        groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=0.3
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, vector_store):
    # Perform similarity search
    docs = vector_store.similarity_search(user_question)

    # Get the conversational chain
    chain = get_conversational_chain()

    # Generate the response
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    return response["output_text"]

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

def main():
    st.set_page_config(page_icon="💬", page_title="McMaster MEST Info Bot")
    st.header("McMaster University (MEST) AI Chatbot")

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I assist you today with information about McMaster MEST?"}
        ]

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Load the FAISS vector store once and store it in session_state
    if "vector_store" not in st.session_state:
        with st.spinner("Loading FAISS index..."):
            vector_store = load_vector_store()
            if vector_store:
                st.session_state["vector_store"] = vector_store
            else:
                st.stop()  # Stop execution if vector store fails to load

    # Chat input handling
    if prompt := st.chat_input(placeholder="Ask a question about McMaster MEST..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Get the answer from the FAISS index
        response = user_input(prompt, st.session_state["vector_store"])

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
