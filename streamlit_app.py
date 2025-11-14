import os
# Set tokenizers parallelism before any other imports to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from rag_system import generate_answer, generate_ui_content

# Initialize chat history and document states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_references" not in st.session_state:
    st.session_state.current_references = []

# Generate dynamic UI content based on the PDF
if "ui_content" not in st.session_state:
    with st.spinner("Initializing..."):
        st.session_state.ui_content = generate_ui_content()

ui = st.session_state.ui_content

# Main chat interface
st.title(ui["title"])
st.write(ui["subtitle"])

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(ui["chat_placeholder"]):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, references = generate_answer(prompt, return_sources=True)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                # Update current references
                st.session_state.current_references = references
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Sidebar content
with st.sidebar:
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.current_references = []
        st.rerun()

    st.markdown("""
    ### About
    """ + ui["about_text"] + """
    
    ### Tips
    - Ask specific questions
    - You can ask follow-up questions
    - Clear the chat history using the button above
    """)

    # Example Questions with clickable functionality
    st.markdown("### Example Questions")
    for question in ui["example_questions"]:
        if st.button(question):
            # Simulate clicking the question
            st.session_state.messages.append({"role": "user", "content": question})
            try:
                response, references = generate_answer(question, return_sources=True)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                st.session_state.current_references = references
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Referenced Documents Section
    st.markdown("### Current References")
    if st.session_state.current_references:
        for i, ref in enumerate(st.session_state.current_references, 1):
            with st.expander(f"Reference {i}"):
                st.markdown(f"```\n{ref}\n```")
                st.markdown("---")
                if hasattr(ref, 'metadata'):
                    st.markdown(f"**Source**: {ref.metadata.get('source', ui['pdf_filename'])}")
                    st.markdown(f"**Section**: {ref.metadata.get('section', 'N/A')}")
    else:
        st.info(ui["references_info"])

    # Document Overview
    st.markdown("### Document Overview")
    with st.expander("📚 Document Information"):
        st.markdown(f"""
        - **{ui['doc_name']}** (Primary Document)
            - {ui['doc_overview']}
        """)
