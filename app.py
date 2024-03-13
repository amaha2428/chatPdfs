import streamlit as st
from functions import get_pdf_text,get_text_chunks,vector_db,user_question

st.header("Chat with Multiple PDF's")

question = st.text_input('Ask a question about your document')

if 'vector' not in st.session_state:
    st.session_state.vector = None

if question and st.session_state.vector is not None:
    chain = user_question(st.session_state.vector)
    response = chain(question)
    st.write(response['result'])

with st.sidebar:
    st.subheader("Follow the outlined process:")
    st.write("1. Upload PDF's by clicking on the 'browse files' button.\n2. Click on the process button  to train the model\n3. You can now ask the PDF's questions ")

    pdf_file = st.file_uploader('Upload only pdf files here', accept_multiple_files=True)

    if st.button("Process"):

        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_file)
            chunks = get_text_chunks(raw_text)
            vector = vector_db(chunks)
            st.session_state.vector = vector
            st.write('build complete!!!')
    
            





 