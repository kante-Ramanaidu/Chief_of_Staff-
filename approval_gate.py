import os
import json
from datetime import datetime
import streamlit as st
from google import genai
from dotenv import load_dotenv

# Import context builder and draft machine modules
import context_builder
import draft_machine
import engine
from draft_machine import SAMPLE_THREADS, draft_reply

# Load environment variables
load_dotenv()

# Set up Streamlit page config
st.set_page_config(
    page_title="AI Email Reply Approval Gate",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Dark Theme & Nice Elements
st.markdown(
    """
    <style>
    /* Dark Theme background */
    .stApp {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161625 !important;
        border-right: 1px solid #2d2d44;
    }
    
    /* Thread boxes */
    .thread-box {
        background-color: #162447;
        border: 1px solid #1f4068;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
    }
    
    .thread-sender {
        font-weight: bold;
        color: #00bbee;
        font-size: 0.95em;
    }
    
    .thread-date {
        font-size: 0.8em;
        color: #8888aa;
        float: right;
    }
    
    .thread-body {
        margin-top: 8px;
        white-space: pre-wrap;
        font-size: 0.9em;
        color: #e0e0e0;
    }
    
    /* Draft Display */
    .draft-container {
        background-color: #1f4068;
        border: 2px solid #00bbee;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        font-size: 1.05em;
        line-height: 1.5;
        white-space: pre-wrap;
        color: #ffffff;
    }
    
    /* Status indicators */
    .status-approved {
        background-color: #1b5e20;
        border: 1px solid #4caf50;
        color: #e8f5e9;
        padding: 12px;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .status-rejected {
        background-color: #b71c1c;
        border: 1px solid #f44336;
        color: #ffebee;
        padding: 12px;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .metadata-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        margin-right: 5px;
        margin-top: 5px;
    }
    .badge-priority-urgent { background-color: #e94560; color: #fff; }
    .badge-priority-needs-reply { background-color: #f0a500; color: #fff; }
    .badge-priority-fyi { background-color: #0f4c75; color: #fff; }
    .badge-category { background-color: #3282b8; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Monkey-patch get_thread_history and get_past_replies to use mock data when running with Sample/Custom Threads ---
def mock_get_thread_history(thread_id: str):
    # Find in session state or default SAMPLE_THREADS
    for t in SAMPLE_THREADS:
        if t["thread_id"] == thread_id:
            return t.get("history", [])
    
    # Check if we have a custom thread currently in session_state
    if "custom_thread_data" in st.session_state and st.session_state.custom_thread_data:
        if st.session_state.custom_thread_data.get("thread_id") == thread_id:
            return st.session_state.custom_thread_data.get("history", [
                {
                    "sender": st.session_state.custom_thread_data.get("sender", "unknown"),
                    "date": st.session_state.custom_thread_data.get("date", "unknown"),
                    "content": st.session_state.custom_thread_data.get("snippet", "")
                }
            ])
            
    return []

def mock_get_past_replies(limit: int = 3):
    return [
        {
            "subject": "Re: Project proposal",
            "content": "Let's proceed with the phase 1 rollout. I'll review the budget by tomorrow."
        },
        {
            "subject": "Re: Meeting schedule",
            "content": "Thanks for confirming. I'm available at 3 PM as proposed."
        },
        {
            "subject": "Re: Quick Sync",
            "content": "Sounds good. Let's touch base on Thursday afternoon."
        }
    ]

# Apply monkey patching so context_builder doesn't hit Gmail API during demo
engine.get_thread_history = mock_get_thread_history
engine.get_past_replies = mock_get_past_replies
context_builder.get_thread_history = mock_get_thread_history
context_builder.get_past_replies = mock_get_past_replies

# --- State Management Initialization ---
if "current_draft" not in st.session_state:
    st.session_state.current_draft = None
if "status" not in st.session_state:
    st.session_state.status = "none" # "none", "approved", "editing", "rejected"
if "generation_count" not in st.session_state:
    st.session_state.generation_count = 0
if "edited_draft_text" not in st.session_state:
    st.session_state.edited_draft_text = ""
if "selected_thread_id" not in st.session_state:
    st.session_state.selected_thread_id = SAMPLE_THREADS[0]["thread_id"]
if "custom_thread_json" not in st.session_state:
    st.session_state.custom_thread_json = ""
if "custom_thread_data" not in st.session_state:
    st.session_state.custom_thread_data = None

# --- API Key Management ---
api_key = os.getenv("GEMINI_API_KEY")

st.sidebar.title("Configuration")

if not api_key:
    api_key_input = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if api_key_input:
        api_key = api_key_input
        # Update draft_machine and gemini client with user-provided key
        os.environ["GEMINI_API_KEY"] = api_key
        draft_machine.GEMINI_API_KEY = api_key
        draft_machine.client = genai.Client(api_key=api_key)
    else:
        st.sidebar.warning("Please enter a Gemini API Key to proceed.")
else:
    # Key is present, make sure client is initialized
    draft_machine.GEMINI_API_KEY = api_key
    draft_machine.client = genai.Client(api_key=api_key)

# --- Sidebar Selection ---
st.sidebar.header("Thread Selection")

# Dropdown for sample threads
thread_options = {t["subject"]: t["thread_id"] for t in SAMPLE_THREADS}
selected_option = st.sidebar.selectbox(
    "Select Sample Thread",
    options=list(thread_options.keys()),
    index=0
)

active_thread_id = thread_options[selected_option]

# Check if selected thread has changed to reset draft states if needed
if active_thread_id != st.session_state.selected_thread_id:
    st.session_state.selected_thread_id = active_thread_id
    st.session_state.current_draft = None
    st.session_state.status = "none"
    st.session_state.generation_count = 0

# Custom Thread JSON Input
st.sidebar.subheader("Or Paste Custom Thread JSON")
custom_json = st.sidebar.text_area(
    "Custom Thread JSON",
    value=st.session_state.custom_thread_json,
    height=200,
    help="Paste thread JSON matching context_builder structure."
)

current_thread = None

if custom_json:
    try:
        parsed_custom = json.loads(custom_json)
        # Ensure it has thread_id or assign a generic one
        if "thread_id" not in parsed_custom:
            parsed_custom["thread_id"] = "custom_thread"
        
        st.session_state.custom_thread_data = parsed_custom
        st.session_state.custom_thread_json = custom_json
        current_thread = parsed_custom
        st.sidebar.success("Custom thread parsed successfully!")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON format.")
        current_thread = next(t for t in SAMPLE_THREADS if t["thread_id"] == active_thread_id)
else:
    st.session_state.custom_thread_data = None
    st.session_state.custom_thread_json = ""
    current_thread = next(t for t in SAMPLE_THREADS if t["thread_id"] == active_thread_id)

# Generate Draft Button
generate_clicked = st.sidebar.button("Generate Draft", use_container_width=True)

if generate_clicked:
    if not api_key:
        st.sidebar.error("Gemini API Key is required to generate a draft!")
    else:
        with st.spinner("Generating AI reply draft..."):
            try:
                # Call draft generation
                draft_text = draft_reply(current_thread)
                st.session_state.current_draft = draft_text
                st.session_state.status = "none"
                st.session_state.generation_count += 1
            except Exception as e:
                st.error(f"Error generating draft: {e}")

# --- Main App Interface ---
st.title("✍️ AI Email Ghostwriter - Approval Gate")
st.write("Review, edit, and approve email reply drafts. **Human-in-the-loop: AI proposes, you authorize.**")

col1, col2 = st.columns(2)

# --- Left Column: Thread History ---
with col1:
    st.subheader("📬 Thread History")
    
    # Display details of the active thread
    priority = current_thread.get("priority", "needs reply")
    category = current_thread.get("category", "project")
    reason = current_thread.get("reason", "No reason provided")
    
    st.markdown(f"**Subject:** {current_thread.get('subject', 'No Subject')}")
    st.markdown(f"**To/From:** {current_thread.get('sender', 'Unknown')}")
    
    priority_class = f"badge-priority-{priority.replace(' ', '-')}"
    st.markdown(
        f'<span class="metadata-badge {priority_class}">Priority: {priority.upper()}</span>'
        f'<span class="metadata-badge badge-category">Category: {category.upper()}</span>',
        unsafe_allow_html=True
    )
    st.markdown(f"*Classification Reason: {reason}*")
    st.write("---")
    
    # Thread history messages
    history_messages = current_thread.get("history", [
        {
            "sender": current_thread.get("sender", "Unknown"),
            "date": current_thread.get("date", "Unknown date"),
            "content": current_thread.get("snippet", "(No snippet or body text)")
        }
    ])
    
    for msg in history_messages:
        st.markdown(
            f"""
            <div class="thread-box">
                <span class="thread-sender">{msg.get('sender')}</span>
                <span class="thread-date">{msg.get('date')}</span>
                <div class="thread-body">{msg.get('content')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Right Column: Draft & Actions ---
with col2:
    st.subheader("🤖 Generated Draft")
    
    if st.session_state.current_draft is None:
        st.info("No draft generated yet. Click **'Generate Draft'** in the sidebar to begin.")
    else:
        # Show generation details
        st.markdown(f"*Generation Count for this thread:* `{st.session_state.generation_count}`")
        
        # Display draft based on current status
        if st.session_state.status == "approved":
            st.markdown('<div class="status-approved">✅ Draft Approved! Ready to send.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="draft-container">{st.session_state.current_draft}</div>', unsafe_allow_html=True)
            
        elif st.session_state.status == "rejected":
            st.markdown('<div class="status-rejected">❌ Draft Rejected. Feel free to regenerate or edit.</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="opacity: 0.5;" class="draft-container">{st.session_state.current_draft}</div>', unsafe_allow_html=True)
            
        elif st.session_state.status == "editing":
            st.markdown("📝 **Editing Draft**")
            # Set text area with the draft text or previously edited text
            edited_text = st.text_area(
                "Modify reply text:",
                value=st.session_state.edited_draft_text or st.session_state.current_draft,
                height=250
            )
            st.session_state.edited_draft_text = edited_text
            
            # Sub-approve button specifically for edited text
            if st.button("Save & Approve Edited Version", type="primary"):
                st.session_state.current_draft = edited_text
                st.session_state.status = "approved"
                
                # Save approved draft to approved_drafts.json
                approved_item = {
                    "thread_id": current_thread.get("thread_id", "unknown"),
                    "subject": current_thread.get("subject", ""),
                    "sender": current_thread.get("sender", ""),
                    "draft": edited_text,
                    "approved_at": datetime.now().isoformat(),
                    "edited": True
                }
                
                # File save handling
                try:
                    if os.path.exists("approved_drafts.json"):
                        with open("approved_drafts.json", "r", encoding="utf-8") as f:
                            approved_list = json.load(f)
                    else:
                        approved_list = []
                    
                    approved_list.append(approved_item)
                    
                    with open("approved_drafts.json", "w", encoding="utf-8") as f:
                        json.dump(approved_list, f, indent=4)
                except Exception as e:
                    st.error(f"Error saving to approved_drafts.json: {e}")
                
                st.rerun()
                
            if st.button("Cancel Editing"):
                st.session_state.status = "none"
                st.session_state.edited_draft_text = ""
                st.rerun()
                
        else:
            # Normal 'none' status - show draft + the three action buttons
            st.markdown(f'<div class="draft-container">{st.session_state.current_draft}</div>', unsafe_allow_html=True)
            
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            # APPROVE action
            with btn_col1:
                if st.button("👍 APPROVE", use_container_width=True, type="primary"):
                    st.session_state.status = "approved"
                    
                    approved_item = {
                        "thread_id": current_thread.get("thread_id", "unknown"),
                        "subject": current_thread.get("subject", ""),
                        "sender": current_thread.get("sender", ""),
                        "draft": st.session_state.current_draft,
                        "approved_at": datetime.now().isoformat(),
                        "edited": False
                    }
                    
                    try:
                        if os.path.exists("approved_drafts.json"):
                            with open("approved_drafts.json", "r", encoding="utf-8") as f:
                                approved_list = json.load(f)
                        else:
                            approved_list = []
                        
                        approved_list.append(approved_item)
                        
                        with open("approved_drafts.json", "w", encoding="utf-8") as f:
                            json.dump(approved_list, f, indent=4)
                    except Exception as e:
                        st.error(f"Error saving to approved_drafts.json: {e}")
                    
                    st.rerun()
                    
            # EDIT action
            with btn_col2:
                if st.button("📝 EDIT", use_container_width=True):
                    st.session_state.status = "editing"
                    st.session_state.edited_draft_text = st.session_state.current_draft
                    st.rerun()
                    
            # REJECT action
            with btn_col3:
                if st.button("👎 REJECT", use_container_width=True):
                    st.session_state.status = "rejected"
                    st.rerun()
