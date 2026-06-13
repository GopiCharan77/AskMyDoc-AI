import streamlit as st
from models.LLM import llm
from tools.index_tool import indexer
from graph import workflow_compiler

st.set_page_config(page_title="AskMyDoc AI", layout="centered")

st.markdown(
    """<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0F0F1A;
    }
    [data-testid="stHeader"] {
        background-color: transparent;
    }
    </style>""",
    unsafe_allow_html=True,
)


st.markdown(
    "<h1 style='text-align: center; color: #6C63FF;'>AskMyDoc AI</h1>",
    unsafe_allow_html=True,
)

indexed = False
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file:
    temp_file = f"./{uploaded_file.name}"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name
    indexer(temp_file)
    app = workflow_compiler()
    indexed = True


def generate_llm_response(input_text):
    # Generator expression to yield string chunks from the LLM
    st.write_stream(
        block["text"] for chunk in llm.stream(input_text)
        if isinstance(chunk.content, list)
        for block in chunk.content
        if block.get("type") == "text" and block.get("text")
    )


def generate_rag_response(input_text):
    input_dict = {"question": str(input_text)}
    response = app.invoke(input_dict)
    
    # st.write_stream natively accepts generators
    # This streams the text much faster and smoother than updating an info container
    st.write_stream(response["generation"])
    
    # Process and append sources below the streamed response
    ans = "\n\n**Sources:**\n"
    for j, i in enumerate(response["documents"]):
        s = str(i.page_content).replace("\n", " ")
        ans += f"\n{j+1}. "
        if len(s) > 100:
            ans += f"**Document:** {s[:45]}..........{s[-45:]} "
        else:
            ans += f"**Document:** {s} "
        ans += f"**Source:** {i.metadata.get('source', 'Unknown')} "
        if "page" in i.metadata:
            ans += f"**Page:** {int(i.metadata['page'])+1} "
    
    st.info(ans)

    # Show which source was used
    sources = [str(doc.metadata.get("source", "")) for doc in response["documents"]]
    if any("http" in s for s in sources):
        st.info("🌐 Answer sourced from **web search** (document chunks were not relevant enough)")
    else:
        st.success("📄 Answer sourced from your **uploaded document**")


if st.button("🔄 Clear Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

with st.form("my_form"):
    text = st.text_area("Enter text:", "How can I help you?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        if indexed:
            generate_rag_response(text)
        else:
            generate_llm_response(text)
