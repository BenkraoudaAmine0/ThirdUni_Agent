"""
Streamlit Web Interface for AI News Agent
Interactive real-time research with Google Gemini, DuckDuckGo & LangGraph memory.
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from agent import build_news_agent, run_agent_turn, extract_message_text

# Page Configuration
st.set_page_config(
    page_title="AI News Agent | Intelligent Tech Briefing",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #4285F4, #9B51E0, #FF6F61);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: rgba(66, 133, 244, 0.12);
        color: #4285F4;
        font-weight: 600;
        font-size: 0.82rem;
        margin-right: 8px;
    }
    .topic-chip {
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state or "memory" not in st.session_state:
    try:
        agent, memory = build_news_agent()
        st.session_state.agent = agent
        st.session_state.memory = memory
    except Exception as e:
        st.session_state.agent = None
        st.session_state.memory = None
        st.session_state.init_error = str(e)

# Sidebar
with st.sidebar:
    st.title("⚙️ Agent Controls")
    
    # API Key Handling
    env_api_key = os.getenv("GOOGLE_API_KEY")
    if env_api_key:
        st.success("API Key loaded from `.env`", icon="🔑")
    else:
        user_key = st.text_input("Enter Google Gemini API Key", type="password", help="Get a free key from Google AI Studio")
        if user_key:
            try:
                agent, memory = build_news_agent(api_key=user_key)
                st.session_state.agent = agent
                st.session_state.memory = memory
                st.session_state.init_error = None
                st.success("API Key applied!", icon="✅")
            except Exception as e:
                st.error(f"Error initializing agent: {e}")

    st.markdown("---")
    
    # Capabilities Overview
    st.markdown("### 🧩 Architecture")
    st.markdown("""
    - **LLM**: Google Gemini 2.5 Flash
    - **Engine**: LangGraph ReAct Agent
    - **Tool**: DuckDuckGo Web Search
    - **Memory**: Stateful LangGraph `MemorySaver`
    - **Persona**: Curated Tech Journalist Context
    """)

    st.markdown("---")
    st.markdown("### 🧠 Session Context")
    st.caption(f"**Thread ID:** `{st.session_state.thread_id[:12]}...`")
    st.caption(f"**Messages in Memory:** {len(st.session_state.messages)}")

    if st.button("🔄 Reset Conversation", use_container_width=True, help="Clear memory and start a new chat thread"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        # Re-build agent checkpointer for a clean slate
        try:
            agent, memory = build_news_agent()
            st.session_state.agent = agent
            st.session_state.memory = memory
        except Exception:
            pass
        st.rerun()

# Main Header
col_header, col_badge = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">📰 AI News Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Real-time artificial intelligence intelligence briefing with conversational memory.</div>',
        unsafe_allow_html=True
    )
with col_badge:
    st.write("")
    st.markdown('<div style="text-align: right;"><span class="metric-badge">🟢 Live Web Search</span><span class="metric-badge">🧠 Stateful Memory</span></div>', unsafe_allow_html=True)

# Quick Prompt Suggestions (shown when chat is empty)
if not st.session_state.messages:
    st.markdown("#### 🚀 Quick-start Topics")
    col1, col2, col3 = st.columns(3)
    quick_query = None
    with col1:
        if st.button("🔥 Latest AI Announcements This Week", use_container_width=True):
            quick_query = "What are the latest major announcements and model releases in generative AI this week?"
    with col2:
        if st.button("⚡ Open-Source LLM Breakthroughs", use_container_width=True):
            quick_query = "What are the most notable open-source LLM releases and benchmarks recently?"
    with col3:
        if st.button("🤖 AI Hardware & Chip Developments", use_container_width=True):
            quick_query = "What is the latest news regarding AI chips, GPUs, and specialized hardware?"

    if quick_query:
        st.session_state.messages.append({"role": "user", "content": quick_query})
        st.rerun()

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_steps" in msg and msg["tool_steps"]:
            with st.expander("🔍 Search Steps & Sources", expanded=False):
                for step in msg["tool_steps"]:
                    if step["type"] == "tool_call":
                        st.markdown(f"**Tool Invocation:** `{step['name']}` with query: *{step['args'].get('query', step['args'])}*")
                    elif step["type"] == "tool_result":
                        st.text_area("DuckDuckGo Findings", step["output"], height=100, disabled=True)

# Chat Input & Execution
prompt = st.chat_input("Ask about recent AI breakthroughs, model releases, research...")

# Handle newly added query (from quick prompt button or chat input)
active_query = None
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# If the last message is from the user and needs a response:
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]

    if not st.session_state.agent:
        with st.chat_message("assistant"):
            st.error("Agent is not initialized. Please ensure `GOOGLE_API_KEY` is configured in `.env` or in the sidebar.")
    else:
        with st.chat_message("assistant"):
            tool_steps = []
            
            with st.status("Thinking and searching live news...", expanded=True) as status:
                agent = st.session_state.agent
                thread_id = st.session_state.thread_id
                
                # Stream events
                final_response = ""
                generator = run_agent_turn(agent, user_query, thread_id=thread_id)
                try:
                    while True:
                        event = next(generator)
                        tool_steps.append(event)
                        if event["type"] == "tool_call":
                            query_str = event["args"].get("query", str(event["args"]))
                            status.write(f"🔎 **Searching DuckDuckGo:** `{query_str}`")
                        elif event["type"] == "tool_result":
                            status.write("📄 **Retrieved web findings.** Synthesizing briefing...")
                except StopIteration as stop:
                    final_response = stop.value

                status.update(label="Briefing synthesized!", state="complete", expanded=False)

            if not final_response:
                final_response = "I couldn't retrieve a response. Please try rephrasing your query."

            st.markdown(final_response)

            # Record in session messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "tool_steps": tool_steps
            })
            st.rerun()
