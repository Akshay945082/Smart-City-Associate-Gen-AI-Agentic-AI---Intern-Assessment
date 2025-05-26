import streamlit as st
import requests
import os
from typing import Optional, Tuple
import time
from pathlib import Path

# Set page config first
st.set_page_config(page_title="City Info Assistant", page_icon="🗺️", layout="centered")

# Custom CSS for centering and styling
st.markdown("""
    <style>
    .main {
        max-width: 600px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 0.5em;
        font-size: 1.1em;
    }
    .stTextInput>div>div>input {
        text-align: center;
        font-size: 1.1em;
    }
    .stMarkdown {
        text-align: center;
    }
    .stAlert {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Configuration
API_URL = "http://localhost:8000/query"  # Direct URL to the query endpoint
HEALTH_CHECK_URL = "http://localhost:8000/health"  # Direct URL to the health endpoint
REQUEST_TIMEOUT = 60  # 60 seconds timeout

# Helper functions
def check_backend_health() -> bool:
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        st.error(f"Error checking backend health: {str(e)}")
        return False

def query_backend(question: str) -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            json={"query": question},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract data from response
        answer = data.get("answer", "No answer provided")
        confidence = data.get("confidence_score")
        processing_time = data.get("processing_time", time.time() - start_time)
        
        return answer, confidence, processing_time, None
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again.", None, None, "timeout"
    except requests.exceptions.ConnectionError:
        return (
            "Error: Could not connect to the backend service. Please ensure it's running.",
            None, None, "connection"
        )
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}", None, None, "request"
    except Exception as e:
        return f"Unexpected error: {str(e)}", None, None, "unexpected"

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Main UI
st.title("🗺️ City Info Assistant")

# Health check indicator
backend_ok = check_backend_health()
if not backend_ok:
    st.error(
        "⚠️ Backend service is not available. Please ensure the backend is running."
    )
    st.stop()

# Input section
with st.container():
    st.markdown("### Ask a Question")
    query = st.text_input(
        "What would you like to know about the city?",
        placeholder="e.g., What are the main tourist attractions?",
        key="query_input",
        disabled=st.session_state.processing,
        help="Type your question and click Search."
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_clicked = st.button(
            "🔍 Search",
            disabled=st.session_state.processing,
            use_container_width=True,
        )

# Process query
if search_clicked and not st.session_state.processing:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.processing = True
        st.session_state.last_query = query
        with st.spinner("🤔 Thinking..."):
            answer, confidence, processing_time, error_type = query_backend(query)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.append(
                {
                    "question": query,
                    "answer": answer,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "timestamp": timestamp,
                    "error_type": error_type,
                }
            )
        st.session_state.processing = False
        st.rerun()

# Display conversation history
if st.session_state.history:
    st.markdown("### Conversation History")
    for i, chat in enumerate(reversed(st.session_state.history)):
        with st.container():
            st.markdown(f"**Q{i+1}:** {chat['question']}")
            st.markdown(f"**A{i+1}:** {chat['answer']}")
            if chat["confidence"] is not None:
                st.markdown(f"*Confidence: {chat['confidence']:.2f}*")
            if chat["processing_time"] is not None:
                st.markdown(f"*Processed in {chat['processing_time']:.2f}s*")
            st.markdown(f"*{chat['timestamp']}*")
            if chat["error_type"]:
                st.error(f"Backend error: {chat['answer']}")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{i}"):
                    st.session_state.history.pop(len(st.session_state.history) - 1 - i)
                    st.rerun()
            st.markdown("---")

# Add a clear history button
if st.session_state.history:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Clear All History"):
            st.session_state.history = []
            st.rerun()
