import hashlib
import json
import os
from datetime import datetime

import streamlit as st

from models.LLM import llm
from tools.index_tool import indexer
from graph import workflow_compiler
from utils.evaluator import evaluate_answer, regenerate_answer

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

if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "app" not in st.session_state:
    st.session_state.app = None
if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


def stream_text_from_llm(prompt_text: str) -> str:
    def token_stream():
        for chunk in llm.stream(prompt_text):
            content = getattr(chunk, "content", None)

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        yield block["text"]
            elif isinstance(content, str) and content.strip():
                yield content
            elif content is not None:
                yield str(content)

    return st.write_stream(token_stream())


def stream_generation_to_text(generation) -> str:
    def token_stream():
        if isinstance(generation, str):
            yield generation
            return

        for chunk in generation:
            content = getattr(chunk, "content", None)

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        yield block["text"]
            elif isinstance(content, str) and content.strip():
                yield content
            elif content is not None:
                yield str(content)
            else:
                yield str(chunk)

    return st.write_stream(token_stream())


def build_context(documents):
    return "\n\n".join(
        getattr(doc, "page_content", "") for doc in documents if getattr(doc, "page_content", "")
    )


def render_sources(documents):
    ans = "\n\n**Sources:**\n"
    for j, doc in enumerate(documents):
        s = str(doc.page_content).replace("\n", " ")
        ans += f"\n{j+1}. "
        if len(s) > 100:
            ans += f"**Document:** {s[:45]}..........{s[-45:]} "
        else:
            ans += f"**Document:** {s} "
        ans += f"**Source:** {doc.metadata.get('source', 'Unknown')} "
        if "page" in doc.metadata:
            ans += f"**Page:** {int(doc.metadata['page']) + 1} "
    st.info(ans)


def source_label(documents):
    sources = [str(doc.metadata.get("source", "")) for doc in documents]
    return "web" if any("http" in s for s in sources) else "document"


def render_eval_panel(eval_result):
    st.markdown("### Evaluation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faithfulness", f"{eval_result.faithfulness}/100")
    c2.metric("Relevance", f"{eval_result.relevance}/100")
    c3.metric("Context Usage", f"{eval_result.context_usage}/100")
    c4.metric("Overall", f"{eval_result.overall_score}/100")

    st.progress(max(0.0, min(1.0, eval_result.overall_score / 100.0)))
    st.write(f"**Hallucination risk:** {eval_result.hallucination_risk}")
    st.write(f"**Verdict:** {eval_result.verdict}")
    st.caption(eval_result.reasoning)


def log_eval(question, answer, eval_result, mode):
    os.makedirs("logs", exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "mode": mode,
        "faithfulness": eval_result.faithfulness,
        "relevance": eval_result.relevance,
        "context_usage": eval_result.context_usage,
        "hallucination_risk": eval_result.hallucination_risk,
        "overall_score": eval_result.overall_score,
        "verdict": eval_result.verdict,
        "reasoning": eval_result.reasoning,
    }
    with open("logs/rag_evals.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_rag_response(input_text: str):
    app = st.session_state.app
    if app is None:
        st.error("Please upload a PDF first.")
        return

    response = app.invoke({"question": str(input_text)})

    documents = response.get("documents", [])
    context = build_context(documents)

    st.markdown("### Initial Answer")
    initial_answer = stream_generation_to_text(response["generation"])

    eval_result = evaluate_answer(input_text, context, initial_answer)
    final_answer = initial_answer
    final_eval = eval_result

    if eval_result.needs_retry or eval_result.overall_score < 75:
        st.warning("The first answer was weak, so the app is regenerating once.")
        regenerated_answer = regenerate_answer(
            input_text,
            context,
            initial_answer,
            eval_result.reasoning,
        )
        st.markdown("### Regenerated Answer")
        st.write(regenerated_answer)
        regenerated_eval = evaluate_answer(input_text, context, regenerated_answer)

        if regenerated_eval.overall_score >= eval_result.overall_score:
            final_answer = regenerated_answer
            final_eval = regenerated_eval
            st.success("Regenerated answer selected.")
        else:
            st.info("The regenerated answer did not improve the score, so the initial answer is kept.")
    else:
        st.success("The answer passed evaluation on the first pass.")

    render_eval_panel(final_eval)
    render_sources(documents)

    mode = source_label(documents)
    if mode == "web":
        st.info("🌐 Answer sourced from **web search** (document chunks were not relevant enough)")
    else:
        st.success("📄 Answer sourced from your **uploaded document**")

    log_eval(input_text, final_answer, final_eval, mode)


uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.uploaded_hash != file_hash:
        temp_dir = "./uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_file, "wb") as file:
            file.write(file_bytes)

        indexer(temp_file)
        st.session_state.app = workflow_compiler()
        st.session_state.indexed = True
        st.session_state.uploaded_hash = file_hash
        st.session_state.uploaded_name = uploaded_file.name
        st.success(f"Indexed: {uploaded_file.name}")

if st.button("🔄 Clear Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

with st.form("my_form"):
    text = st.text_area("Enter text:", "How can I help you?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        if st.session_state.indexed and st.session_state.app is not None:
            generate_rag_response(text)
        else:
            generate_llm_response(text)
