import streamlit as st
import time

from utils import process_image, process_audio
from rag_engine import init_vector_store, save_full_memory_trace

from agents.parser import run_parser_agent
from agents.router import run_router_agent
from agents.solver import run_solver_agent
from agents.verifier import run_verifier_agent
from agents.explainer import run_explainer_agent

st.set_page_config(
    page_title="Math Mentor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stStatus { border-left: 4px solid #4CAF50; background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .stTextArea textarea { font-size: 16px; }
    .agent-box { border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    div.stButton > button:first-child { border-radius: 8px; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

if "step" not in st.session_state:
    st.session_state.step = 1
if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = {}
if "raw_input" not in st.session_state:
    st.session_state.raw_input = ""
if "logs" not in st.session_state:
    st.session_state.logs = []

with st.sidebar:
    st.title("⚙️ System Internals")
    st.markdown("Live trace of agent activities:")
    log_container = st.container()

    def add_log(message):
        st.session_state.logs.append(message)
        with log_container:
            for log in st.session_state.logs[-10:]:  # Show last 10 logs
                st.caption(log)


st.title("🧠 Transparent Multimodal Math Mentor")

if st.session_state.step == 1:
    st.info("👋 Welcome! Upload an image, speak, or type a math problem.")

    tabs = st.tabs(["📝 Text", "📷 Image", "🎙️ Audio"])

    captured_text = ""

    with tabs[0]:
        captured_text = st.text_area(
            "Enter problem:",
            height=150,
            placeholder="e.g., A box contains 4 red balls...",
        )

    with tabs[1]:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
        if uploaded_file:
            st.image(uploaded_file, width=300)
            if st.button("Extract Text"):
                with st.spinner("👀 OCR Agent scanning..."):
                    text, err = process_image(uploaded_file)
                    if not err:
                        captured_text = text
                        add_log("✅ OCR Agent: Image processed successfully")
                    else:
                        st.error(f"OCR Error: {err}")

    with tabs[2]:
        audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
        if audio_file:
            if st.button("Transcribe"):
                with st.spinner("👂 Audio Agent listening..."):
                    # Save temp file for whisper
                    with open("temp.mp3", "wb") as f:
                        f.write(audio_file.getbuffer())
                    text, err = process_audio("temp.mp3")
                    if not err:
                        captured_text = text
                        add_log("✅ Audio Agent: Transcription complete")
                    else:
                        st.error(f"Audio Error: {err}")

    # Processing Trigger
    if captured_text or st.session_state.raw_input:
        if captured_text:
            st.session_state.raw_input = captured_text

        st.write("---")
        st.markdown(f"**Detected Input:** `{st.session_state.raw_input}`")

        if st.button("Analyze Problem 🚀", use_container_width=True):
            # FANCY LOADING SCREEN START
            with st.status("🤖 Orchestrating Agents...", expanded=True) as status:
                st.write("📝 **Parser Agent**: Analyzing text structure...")
                time.sleep(0.5)  # Fake delay for visual effect
                st.session_state.parsed_data = run_parser_agent(
                    st.session_state.raw_input
                )
                add_log(f"✅ Parser: Extracted problem")
                status.update(
                    label="Parsing Complete!", state="complete", expanded=False
                )

            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    st.subheader("🕵️ User Verification (HITL)")

    current_text = st.session_state.parsed_data.get("problem_text", "")

    col1, col2 = st.columns([2, 1])
    with col1:
        edited_text = st.text_area(
            "Verify & Edit the problem if needed:", value=current_text, height=200
        )

    with col2:
        st.markdown("### Why this step?")
        st.caption(
            "AI models can misinterpret OCR or context. Your verification ensures 100% accuracy before solving."
        )

        if st.button("✅ Confirm & Solve", type="primary", use_container_width=True):
            st.session_state.parsed_data["problem_text"] = edited_text
            add_log("👤 User: Confirmed problem text")
            st.session_state.step = 3
            st.rerun()

        if st.button("❌ Restart", use_container_width=True):
            st.session_state.step = 1
            st.session_state.raw_input = ""
            st.rerun()

elif st.session_state.step == 3:
    problem = st.session_state.parsed_data["problem_text"]

    # --- PHASE 1: MAIN SOLVER (Fast) ---
    # We use a status container for the critical path only
    with st.status("🧠 Orchestrating Agents...", expanded=True) as status:
        # 1. Memory Init
        st.write("📚 **Memory Agent**: Accessing Knowledge Base...")
        init_vector_store()

        # 2. Router
        st.write("🔀 **Router Agent**: Analyzing complexity...")
        topic = run_router_agent(problem)
        st.info(f"👉 Routing to **{topic}** Specialist")
        time.sleep(0.3)  # UI Pacing

        # 3. Solver
        st.write(f"💡 **Solver Agent**: Computing answer...")
        solution, context = run_solver_agent(problem, topic)

        status.update(label="✅ Solution Generated!", state="complete", expanded=False)

    # --- PHASE 2: DISPLAY SOLUTION (Immediate) ---
    col_main, col_details = st.columns([2, 1])

    with col_main:
        st.subheader("📝 Final Solution")
        st.success(solution)

        # --- PHASE 3: SECONDARY AGENTS (Lazy Loading) ---
        # The user sees the solution above immediately.
        # The expander below will show a loading state while the extra agents run.

        with st.expander("🔎 View Verification & Explanation (Analysis running...)"):
            # 1. VERIFIER LOADING STATE
            st.markdown("### 🛡️ Verifier Report")
            verify_placeholder = st.empty()  # Create a placeholder

            with verify_placeholder.container():
                with st.spinner("🕵️ Verifier Agent is checking logic..."):
                    # This runs while the spinner is visible
                    verification_result = run_verifier_agent(problem, solution)

            # Update placeholder with final result
            verify_placeholder.write(verification_result)

            st.divider()

            # 2. EXPLAINER LOADING STATE
            st.markdown("### 👨‍🏫 Teacher Explanation")
            explain_placeholder = st.empty()

            with explain_placeholder.container():
                with st.spinner("🎓 Explainer Agent is drafting notes..."):
                    explanation_result = run_explainer_agent(problem, solution)

            # Update placeholder
            explain_placeholder.write(explanation_result)

    # --- RIGHT PANEL: DEBUG INFO ---
    with col_details:
        st.subheader("📊 Transparency")
        st.markdown("**Context Retrieved:**")
        if context:
            for c in context:
                st.caption(f"📄 {c[:80]}...")
        else:
            st.caption("No RAG context needed.")

        st.markdown("**Agent Latency:**")
        st.json(
            {
                "Topic": topic,
                "Solver": "Active",
                "Verifier": "Complete",
                "Explainer": "Complete",
            }
        )

    # --- PHASE 4: MEMORY & FEEDBACK LOOP ---
    st.divider()
    st.subheader("🧠 Self-Learning Feedback")
    st.write("Help the AI get smarter. Is this solution correct?")

    col_feed1, col_feed2 = st.columns(2)

    with col_feed1:
        if st.button("👍 Yes, Correct (Save Pattern)"):
            # Prepare rich memory packet
            memory_packet = {
                "original_input_type": "Text/Image",  # Simplified for demo
                "parsed_question": problem,
                "topic": topic,
                "retrieved_context": context,
                "final_answer": solution,
                "verifier_outcome": verification_result,
                "user_feedback": "positive",
            }

            # Save to JSON
            save_full_memory_trace(memory_packet)

            st.success(
                "✅ Pattern memorized! The system will use this logic for similar future problems."
            )
            time.sleep(2)
            st.session_state.step = 1
            st.session_state.raw_input = ""
            st.rerun()

    with col_feed2:
        if st.button("👎 No, Incorrect (Discard)"):
            st.warning(
                "❌ Feedback noted. This solution will NOT be added to long-term memory."
            )
            time.sleep(2)
            st.session_state.step = 1
            st.session_state.raw_input = ""
            st.rerun()

    st.write("---")
    if st.button("🔄 Solve Another Problem (No Save)", use_container_width=True):
        st.session_state.step = 1
        st.session_state.raw_input = ""
        st.rerun()
