"""
AI News Agent Core Module
Powered by LangGraph, Google Gemini, and DuckDuckGo Search.
Includes conversational memory and tech-journalist context.
"""

import os
from typing import Generator, Dict, Any, Optional
from dotenv import load_dotenv

# Suppress minor deprecation warnings for cleaner CLI and Streamlit logs
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Ensure .env is loaded
load_dotenv()

DEFAULT_SYSTEM_PROMPT = """You are an expert AI & Tech Journalist Assistant.
Your mission is to provide accurate, up-to-date news, breakthroughs, and analysis across Artificial Intelligence, LLMs, Machine Learning, and tech innovation.

Guidelines:
1. Recency & Verification: For questions about recent news, releases, announcements, or benchmarks, ALWAYS use the DuckDuckGo search tool to retrieve the latest verified facts.
2. Formatted Output:
   - Executive Summary: A crisp 1-2 sentence overview of the news.
   - Key Highlights: Bullet points covering what happened, key players, specs, or impact.
   - Sources & Dates: Clearly cite dates and source outlets/domains found in search.
3. Conversational Memory: Seamlessly refer back to previously discussed models, companies, or topics in the chat.
4. Professional Tone: Maintain an objective, engaging, and tech-savvy journalistic voice.
"""

DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"


def extract_message_text(content: Any) -> str:
    """Safely extract plain text from LangChain message content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content) if content is not None else ""


def build_news_agent(
    api_key: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    checkpointer: Optional[MemorySaver] = None
):

    # 1. Look for manual input, then Streamlit Secrets, then local environment variables
    import streamlit as st
    
    key = api_key
    if not key and "GOOGLE_API_KEY" in st.secrets:
        key = st.secrets["GOOGLE_API_KEY"]
    if not key:
        key = os.getenv("GOOGLE_API_KEY")
        
    if not key:
        raise ValueError(
            "Google API Key not found. Please set GOOGLE_API_KEY in your secrets "
            "or provide it through the interface."
        )
    
    """
    Builds and returns a compiled LangGraph ReAct agent with:
    - Google Gemini Chat model
    - DuckDuckGo web search tool
    - Tech journalist persona system context
    - MemorySaver checkpointing for multi-turn conversational memory
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Google API Key not found. Please set GOOGLE_API_KEY in your .env file "
            "or provide it through the interface."
        )

    # Initialize Gemini Model
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL_NAME,
        temperature=0.3,
        google_api_key=key,
    )

    # Setup DuckDuckGo Search Tool
    search_tool = DuckDuckGoSearchRun(
        name="duckduckgo_search",
        description="Search DuckDuckGo for the latest AI news, announcements, models, and articles."
    )
    tools = [search_tool]

    # Shared or new in-memory checkpointer
    memory = checkpointer if checkpointer is not None else MemorySaver()

    # Compile the agent with context and memory
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=memory
    )

    return agent, memory


def run_agent_turn(
    agent,
    user_query: str,
    thread_id: str = "default_session"
) -> Generator[Dict[str, Any], None, str]:
    """
    Executes a single agent turn with step-by-step event streaming.
    Yields events:
      - {'type': 'tool_call', 'name': ..., 'args': ...}
      - {'type': 'tool_result', 'output': ...}
    Returns:
      Final synthesized response text.
    """
    config = {"configurable": {"thread_id": thread_id}}
    final_text = ""

    for chunk in agent.stream(
        {"messages": [("user", user_query)]},
        config=config,
        stream_mode="updates"
    ):
        for node_name, node_update in chunk.items():
            messages = node_update.get("messages", [])
            for msg in messages:
                # Check for tool call generation
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        yield {
                            "type": "tool_call",
                            "name": call.get("name", "Search"),
                            "args": call.get("args", {})
                        }

                # Check for tool execution outputs
                if node_name == "tools" or msg.__class__.__name__ == "ToolMessage":
                    yield {
                        "type": "tool_result",
                        "output": extract_message_text(msg.content)
                    }

                # Agent node with text content
                if node_name == "agent" and msg.content:
                    extracted = extract_message_text(msg.content)
                    if extracted.strip():
                        final_text = extracted

    return final_text
