

# Author: Sai Vikhyath Kudhroli
# Date: 8 July 2024


import os
import pandas as pd
import streamlit as st


from QuestionAnsweringBot import question_answering
from GeneratingInsights import generate_insights
from RequirementsSatisfaction import generate_suggestion


def save_uploaded_file(uploaded_file, path):
    """ Documentation goes here """
    save_path = os.path.join(path)
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def delete_files(path):
    """ Documentation goes here """
    try:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        pass


def extract_information():
    """ Documentation goes here """

    st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>Information Extraction</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a file: ", type="pdf")

    st.markdown("<p style='margin-top: 30px; margin-bottom: 1px;'></p>", unsafe_allow_html=True)
    textarea_style = """
        <style>
            .stTextArea textarea{
                padding: 15px;
                padding-left: 20px;
                height: 120px;
            }
        </style>
    """
    st.markdown(textarea_style, unsafe_allow_html=True)
    question = st.text_area("Enter your question: ")
    
    st.markdown(
        """
            <style>
                .stButton button {
                    width: 100%;
                }
            </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("Extract Information"):
        with st.spinner("Extracting information..."):
            delete_files("Documents/QA/NewlyUploaded")
            if uploaded_file is not None and question:
                save_uploaded_file(uploaded_file, "Documents/QA/NewlyUploaded")
                extracted_info = question_answering(question)
                st.session_state.output = extracted_info
                st.success("Extraction completed!!!")
                delete_files("Documents/QA/NewlyUploaded")
            else:
                st.session_state.output = "Please upload a PDF file and enter a question."
    else:
        if 'output' not in st.session_state:
            st.session_state.output = "You'll see the output here"

    st.markdown("")
    st.markdown(
        f"""
            <div style="border: 10px solid #e6e6e6; padding: 30px; border-radius: 5px;">
                {st.session_state.output}
            </div>
        """,
        unsafe_allow_html=True
    )


def extract_insights():
    """ Documentation goes here """
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>Generate Insights</h1>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload files: ", type="pdf", accept_multiple_files=True)
 
    st.markdown(
        """
            <style>
                .stButton button {
                    width: 100%;
                }
            </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("Generate Insights"):
        with st.spinner("Generating insights..."):
            delete_files("Documents/Insights/NewlyUploaded")
            if uploaded_files is not None and len(uploaded_files) > 0:
                for uploaded_file in uploaded_files:
                    save_uploaded_file(uploaded_file, "Documents/Insights/NewlyUploaded")
                generated_insights = generate_insights()
                st.session_state.output = generated_insights
                st.success("Insights Generated!!!")
                delete_files("Documents/Insights/NewlyUploaded")
            else:
                st.session_state.output = "Please upload at least one PDF file."
    else:
        if 'output' not in st.session_state:
            st.session_state.output = "You'll see the output here"

    st.markdown("")
    if isinstance(st.session_state.output, pd.DataFrame):
        st.markdown("")
        st.dataframe(st.session_state.output, width=1500)
    else:
        st.markdown(
            f"""
                <div style="border: 10px solid #e6e6e6; padding: 30px; border-radius: 5px; text-align: center;">
                    {st.session_state.output}
                </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


def suggest_candidates():
    """ Documentation goes here """
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>Suggest Best Matching Candidates</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload files: ", type="csv")
 
    st.markdown(
        """
            <style>
                .stButton button {
                    width: 100%;
                }
            </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("Compute Best Candidates"):
        with st.spinner("Computing best matching candidates..."):
            delete_files("Documents/RequirementsMatching/Input")
            if uploaded_file is not None:
                save_uploaded_file(uploaded_file, "Documents/RequirementsMatching/Input")
                best_candidates, scores_dataframe = generate_suggestion()
                st.session_state.output = scores_dataframe
                st.session_state.bestCandidates = best_candidates
                st.success("Computed Best Matching Candidate!!!")
                delete_files("Documents/RequirementsMatching/Input")
            else:
                st.session_state.output = "Please upload at least one CSV file."
                st.session_state.bestCandidates = ""
    else:
        if 'output' not in st.session_state:
            st.session_state.output = "You'll see the output here"
        if 'bestCandidates' not in st.session_state:
            st.session_state.bestCandidates = ""

    st.markdown("")
    if isinstance(st.session_state.output, pd.DataFrame):
        st.markdown("")
        st.markdown(f"**Best Candidate(s): {', '.join(st.session_state.bestCandidates)}**")
        st.markdown("Computation Methodology: ")
        st.dataframe(st.session_state.output, width=1500)
    else:
        st.markdown(
            f"""
                <div style="border: 10px solid #e6e6e6; padding: 30px; border-radius: 5px; text-align: center;">
                    {st.session_state.output}
                </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


def home():
    """ Documentation goes here """

    st.title("Welcome to InfoQuest AI")
    st.subheader("Your go-to application for extracting insights and information from documents")

    st.markdown("<hr style='border: 1px solid;'>", unsafe_allow_html=True)

    st.markdown("""
    ## About Us
    InfoQuest AI is an application that leverages Large Language Models to offer insights from documents, built by a research team at Arizona State University.

    ### Our Services
    - **Extract Information:** Upload a PDF document and type in a query to extract the answer for the query.
    - **Extract Insights:** Upload a set of invoice documents and select a set of keywords to extract insights for the selected keywords.
    - **Suggest Candidates:** Upload a CSV document that contains requirements, their weights, and candidates. We'll compute the best candidate for you based on your requirements.
    """)

    st.markdown("<hr style='border: 1px solid;'>", unsafe_allow_html=True)


def main():
    """ Documentation goes here """
    
    st.set_page_config(
        page_title="InfoQuest AI",
        page_icon=":computer:",
        layout="wide",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Home", "Extract Information", "Extract Insights", "Suggest Candidates"])

    if page == "Home":
        home()
    elif page == "Extract Information":
        st.session_state.output = "You'll see the output here"
        extract_information()
    elif page == "Extract Insights":
        st.session_state.output = "You'll see the output here"
        extract_insights()
    elif page == "Suggest Candidates":
        st.session_state.output = "You'll see the output here"
        suggest_candidates()


if __name__ == "__main__":
    main()
