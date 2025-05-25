import streamlit as st
import os
import time
import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from groq import Groq
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# Configure logging with improved format and file handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('application.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger: logging.Logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_TIMEOUT: int = 30
MAX_CHAT_HISTORY: int = 50  # Limit chat history to prevent memory issues
MIN_API_KEY_LENGTH: int = 32  # Minimum length for API key validation
MAX_TOKENS: int = 150
DEFAULT_TEMPERATURE: float = 0.7

# Ensure session state variable is initialized before widget creation
if "quickvibe_response_vibe" not in st.session_state:
    st.session_state["quickvibe_response_vibe"] = "Witty 😏"

# --- Security and Validation Functions ---
def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate API key format and basic security requirements.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key or not isinstance(api_key, str):
        return False, "API key is required"
    
    api_key = api_key.strip()
    
    if len(api_key) < MIN_API_KEY_LENGTH:
        return False, f"API key too short (minimum {MIN_API_KEY_LENGTH} characters)"
    
    # Basic format validation for Groq API keys
    if not re.match(r'^gsk_[a-zA-Z0-9_-]+$', api_key):
        return False, "Invalid API key format (should start with 'gsk_')"
    
    return True, ""

def sanitize_input(input_value: str) -> str:
    """
    Sanitize user input with enhanced validation.
    
    Args:
        input_value: The input string to sanitize
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_value, str):
        return ""
    
    # Remove excessive whitespace and limit length
    sanitized = input_value.strip()[:2000]  # Limit to 2000 characters
    
    # Remove potential harmful patterns (basic XSS prevention)
    sanitized = re.sub(r'<[^>]*>', '', sanitized)
    
    return sanitized

# --- API and Model Management ---
def fetch_groq_models_for_quickvibe(api_key: str) -> Tuple[bool, Union[List[Dict[str, Any]], str]]:
    """
    Fetch available models from Groq API with enhanced error handling.
    
    Args:
        api_key: Valid Groq API key
        
    Returns:
        Tuple of (success, models_list_or_error_message)
    """
    # Validate API key first
    is_valid, error_msg = validate_api_key(api_key)
    if not is_valid:
        return False, f"API key validation failed: {error_msg}"
    
    try:
        client: Groq = Groq(api_key=api_key)
        models_page = client.models.list()
        
        if not hasattr(models_page, 'data') or not models_page.data:
            return False, "No models returned from API"
        
        models_list: List[Dict[str, Any]] = [
            m.model_dump() for m in models_page.data 
            if hasattr(m, 'id') and m.id
        ]
        
        # Filter for chat models with enhanced filtering
        chat_models_list: List[Dict[str, Any]] = [
            m for m in models_list
            if m.get('id') and _is_chat_model(m.get('id', ''))
        ]
        
        if not chat_models_list:
            return False, "No suitable chat models found. QuickVibe needs models like Llama, Mixtral, or Gemma."
        
        logger.info(f"Successfully fetched {len(chat_models_list)} chat models")
        return True, chat_models_list
        
    except Exception as e:
        error_msg = f"Failed to fetch models: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def _is_chat_model(model_id: str) -> bool:
    """
    Determine if a model is suitable for chat based on its ID.
    
    Args:
        model_id: The model identifier
        
    Returns:
        True if the model is suitable for chat
    """
    excluded_keywords: List[str] = [
        'tts', 'whisper', 'vision', 'audio', 'image', 'speech', 
        'scout', 'maverick', 'playai', 'llama-guard', 'guard'
    ]
    
    included_keywords: List[str] = ['llama', 'mixtral', 'gemma']
    
    model_lower = model_id.lower()
    
    # Check for excluded keywords
    if any(keyword in model_lower for keyword in excluded_keywords):
        return False
    
    # Check for included keywords
    return any(keyword in model_lower for keyword in included_keywords)

# --- Session State Management ---
def initialize_quickvibe_session_state() -> None:
    """
    Initialize session state variables with type safety and validation.
    """
    defaults: Dict[str, Any] = {
        'api_key': os.getenv('GROQ_API_KEY', ''),
        'api_validated': False,
        'quickvibe_chat_models': [],
        'current_model_index': 0,
        'chat_history': [],
        'error_log': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Limit chat history size to prevent memory issues
    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
    
    # Auto-validate API key if present
    if st.session_state.api_key and not st.session_state.api_validated:
        _auto_validate_api_key()

def _auto_validate_api_key() -> None:
    """
    Automatically validate API key and fetch models if key is present.
    """
    try:
        success, models_or_error = fetch_groq_models_for_quickvibe(st.session_state.api_key)
        if success and isinstance(models_or_error, list):
            st.session_state.quickvibe_chat_models = [m['id'] for m in models_or_error]
            st.session_state.api_validated = True
            st.session_state.current_model_index = 0
            logger.info("API key auto-validated successfully")
        else:
            st.session_state.api_validated = False
            logger.warning(f"API key auto-validation failed: {models_or_error}")
    except Exception as e:
        logger.error(f"Error during API key auto-validation: {e}")
        st.session_state.api_validated = False

def log_quickvibe_error(error_msg: str) -> None:
    """
    Log error with timestamp and store in session state.
    
    Args:
        error_msg: The error message to log
    """
    logger.error(error_msg)
    
    error_entry: Dict[str, Any] = {
        'timestamp': time.time(),
        'message': error_msg
    }
    
    st.session_state.error_log.append(error_entry)
    
    # Limit error log size
    if len(st.session_state.error_log) > 100:
        st.session_state.error_log = st.session_state.error_log[-50:]

# --- Model Management ---
def _is_quickvibe_api_configured() -> bool:
    """
    Check if API is properly configured and validated.
    
    Returns:
        True if API is ready for use
    """
    return (
        st.session_state.api_validated and 
        bool(st.session_state.quickvibe_chat_models) and
        bool(st.session_state.api_key)
    )

def get_next_quickvibe_model() -> Optional[str]:
    """
    Get the next chat model with improved error handling.
    
    Returns:
        Model ID if available, None otherwise
    """
    if not st.session_state.quickvibe_chat_models:
        logger.warning("No chat models available for rotation")
        return None
    
    try:
        model_id: str = st.session_state.quickvibe_chat_models[st.session_state.current_model_index]
        st.session_state.current_model_index = (
            st.session_state.current_model_index + 1
        ) % len(st.session_state.quickvibe_chat_models)
        
        logger.debug(f"Selected model: {model_id}")
        return model_id
        
    except (IndexError, TypeError) as e:
        logger.error(f"Error rotating models: {e}")
        st.session_state.current_model_index = 0
        return None

# --- Message Processing ---
def get_vibe_instruction(vibe: str) -> str:
    """
    Get instruction text for the selected vibe.
    
    Args:
        vibe: The selected vibe string
        
    Returns:
        Instruction text for the vibe
    """
    vibe_prompts: Dict[str, str] = {
        "Witty 😏": "Be clever, playful, and drop some wordplay or puns.",
        "Savage 🔥": "Be bold, a little ruthless, and don't hold back.",
        "Supportive 🤗": "Be kind, encouraging, and uplifting.",
        "Dry 🧂": "Be deadpan, sarcastic, and a little salty.",
        "Flirty 😘": "Be charming, a little cheeky, and playful.",
        "Chill 😎": "Be relaxed, casual, and super laid-back."
    }
    return vibe_prompts.get(vibe, "Be clever, playful, and drop some wordplay or puns.")

def create_system_prompt(vibe_instruction: str) -> str:
    """
    Create the system prompt with the specified vibe.
    
    Args:
        vibe_instruction: The instruction for the selected vibe
        
    Returns:
        Complete system prompt
    """
    return (
        f"You are QuickVibe AI, a super chill and funny AI that helps craft epic text message responses. "
        f"You talk like a zoomer or gen alpha kid on the internet - use slang, keep it short, be a bit edgy, "
        f"and use emojis. Make the user sound cool and witty. The user will give you a situation or a text "
        f"they received. Give them a few fire response options. Keep it brief, like a text. No cringe, only Ws. "
        f"⚡️😎💯\nVibe: {vibe_instruction}"
    )

def send_quickvibe_message(model: str, user_message: str, temperature: float = DEFAULT_TEMPERATURE) -> Tuple[bool, str]:
    """
    Send message to Groq API with enhanced error handling and validation.
    
    Args:
        model: The model to use
        user_message: The user's message
        temperature: Temperature for response generation
        
    Returns:
        Tuple of (success, response_or_error_message)
    """
    # Input validation
    if not model or not user_message:
        return False, "Model and message are required."
    
    if not st.session_state.api_key:
        return False, "No API key configured. Please add your API key in the sidebar."
    
    # Sanitize input
    clean_message: str = sanitize_input(user_message)
    if not clean_message:
        return False, "Invalid or empty message after sanitization."
    
    # Validate temperature
    temperature = max(0.0, min(2.0, temperature))
    
    try:
        # Get vibe and create prompts
        vibe: str = st.session_state.get("quickvibe_response_vibe", "Witty 😏")
        vibe_instruction: str = get_vibe_instruction(vibe)
        system_prompt: str = create_system_prompt(vibe_instruction)
        
        # Create message payload
        messages_payload: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean_message}
        ]
        
        # Make API call
        client: Groq = Groq(api_key=st.session_state.api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            temperature=temperature,
            max_tokens=MAX_TOKENS
        )
        
        # Extract response
        if completion.choices and completion.choices[0].message.content:
            response_content: str = completion.choices[0].message.content
            logger.info(f"Successful response generated using model: {model}")
            return True, response_content
        else:
            error_msg = "No response content generated"
            log_quickvibe_error(error_msg)
            return False, f"💀 Yikes, got empty response: {error_msg}"
            
    except Exception as e:
        error_msg = f"QuickVibe chat failed with model {model}: {str(e)}"
        log_quickvibe_error(error_msg)
        return False, f"💀 Yikes, server's trippin': {str(e)}"

# --- UI Components ---
def render_api_key_section() -> None:
    """
    Render the API key input and validation section.
    """
    st.header("🔑 API Key Setup")
    st.markdown("Need a Groq API key to get this party started. Grab one from [GroqCloud](https://console.groq.com/keys)!")
    
    api_key_input: str = st.text_input(
        "Groq API Key", 
        type="password", 
        value=st.session_state.api_key,
        key="quickvibe_api_key_input",
        help="Paste your Groq API key here (should start with 'gsk_')"
    ) or ""
    
    # Handle API key changes
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.session_state.api_validated = False
        st.session_state.quickvibe_chat_models = []

    # Connect button
    if st.button("🚀 Connect & Load Models", type="primary"):
        if st.session_state.api_key and st.session_state.api_key.strip():
            with st.spinner("Verifying key & fetching models... ✨"):
                success, models_or_error = fetch_groq_models_for_quickvibe(st.session_state.api_key)
                if success and isinstance(models_or_error, list):
                    st.session_state.quickvibe_chat_models = [m['id'] for m in models_or_error]
                    st.session_state.api_validated = True
                    st.session_state.current_model_index = 0
                    st.success(f"✅ Connected! Found {len(st.session_state.quickvibe_chat_models)} chat models. Let's gooo! 🔥")
                    st.balloons()
                else:
                    st.session_state.api_validated = False
                    st.session_state.quickvibe_chat_models = []
                    st.error(f"❌ Connection Failed: {models_or_error}")
        else:
            st.error("Please enter a valid API key first! 🙄")

def render_api_status() -> None:
    """
    Render the API connection status.
    """
    if st.session_state.api_validated:
        st.success("🟢 API Live & Ready to Vibe!")
        if st.session_state.quickvibe_chat_models:
            st.markdown(f"**Models in rotation:** {len(st.session_state.quickvibe_chat_models)}")
            with st.expander("View Available Models"):
                for i, model in enumerate(st.session_state.quickvibe_chat_models):
                    status = "🎯 Current" if i == st.session_state.current_model_index else "⚪"
                    st.text(f"{status} {model}")
    elif st.session_state.api_key:
        st.warning("🟡 API Key entered, but not validated. Hit that connect button!")
    else:
        st.info("🔴 No API Key. Add one to start vibing.")

def render_vibe_selector() -> None:
    """
    Render the ResponseVibe selection component.
    """
    with st.expander("🎚️ ResponseVibe", expanded=True):
        vibe_options: List[str] = [
            "Witty 😏",
            "Savage 🔥", 
            "Supportive 🤗",
            "Dry 🧂",
            "Flirty 😘",
            "Chill 😎"
        ]
        
        current_vibe: str = st.session_state.get("quickvibe_response_vibe", "Witty 😏")
        selected_index: int = 0
        
        try:
            selected_index = vibe_options.index(current_vibe)
        except ValueError:
            pass  # Use default index if current vibe not found
        
        vibe: str = st.radio(
            "Pick your vibe:",
            vibe_options,
            key="quickvibe_vibe_selector",
            index=selected_index
        )
        
        # Update session state without causing widget conflicts
        if vibe != st.session_state.get("quickvibe_response_vibe"):
            st.session_state["quickvibe_response_vibe"] = vibe

def get_quickvibe_readme() -> str:
    """Returns the README.md content for QuickVibe."""
    return '''
# QuickVibe ⚡️

QuickVibe is a modern, youth-focused AI app for generating instant, high-vibe text message responses. Powered by Groq's blazing-fast Llama/Mixtral models, QuickVibe helps you craft witty, savage, supportive, dry, flirty, or chill replies for any situation.

## Features
- 🔑 Easy Groq API key setup
- ⚡️ Ultra-fast, model-cycled AI responses
- 🎚️ ResponseVibe: Pick your reply style (Witty, Savage, Supportive, Dry, Flirty, Chill)
- 💬 Simple, chat-style interface
- 🧠 Model cycling for variety
- 📝 Copy-paste your texts or situations and get fire responses

## How to Use
1. Get your [Groq API key](https://console.groq.com/keys)
2. Paste it in the sidebar and connect
3. Pick your ResponseVibe
4. Paste a text or describe your situation
5. Get instant, vibey replies!

## Example Vibes
- **Witty 😏**: Clever, playful, wordplay, puns
- **Savage 🔥**: Bold, ruthless, no filter
- **Supportive 🤗**: Kind, encouraging, uplifting
- **Dry 🧂**: Deadpan, sarcastic, salty
- **Flirty 😘**: Charming, cheeky, playful
- **Chill 😎**: Relaxed, casual, laid-back

---
Made for Gen Z/Alpha, meme lords, and anyone who wants to level up their texting game. No cringe, only Ws. ⚡️😎💯
'''

# --- Streamlit App: QuickVibe ---
def quickvibe_main():
    st.set_page_config(
        page_title="QuickVibe ⚡️",
        page_icon="⚡️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for background image and styling
    st.markdown("""
        <style>
        .stApp {
            background-image: url("app/static/logo.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        
        .stApp > header {
            background-color: transparent;
        }

        /* Add a semi-transparent overlay for better readability */
        .main {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
        }

        /* Make text more readable on dark background */
        .stMarkdown, .stText, p, h1, h2, h3 {
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_quickvibe_session_state()
    st.title("QuickVibe ⚡️: Instant Vibe Checks")
    st.caption("Drop the deets & I'll craft some A+ replies 🔥💯")
    
    # Groq Badge
    st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
                <img
                    src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg"
                    alt="Powered by Groq for fast inference."
                    style="height: 40px; width: auto;"
                />
            </a>
        </div>
    """, unsafe_allow_html=True)

    # --- ExampleVibes Button & Modal ---
    if 'show_examplevibes' not in st.session_state:
        st.session_state['show_examplevibes'] = False
    if st.button("✨ exampleVibes", help="See example vibes and app info!"):
        st.session_state['show_examplevibes'] = True
    if st.session_state['show_examplevibes']:
        st.markdown("""
            <style>
            .quickvibe-modal {
                position: fixed;
                top: 10%;
                left: 50%;
                transform: translate(-50%, 0);
                background: #fff;
                border-radius: 16px;
                box-shadow: 0 0 32px #0002;
                padding: 2rem;
                z-index: 9999;
                max-width: 600px;
                width: 90vw;
                max-height: 80vh;
                overflow-y: auto;
            }
            .quickvibe-modal-close {
                position: absolute;
                top: 1rem;
                right: 1.5rem;
                font-size: 1.5rem;
                cursor: pointer;
                color: #888;
            }
            </style>
            <div class="quickvibe-modal">
                <span class="quickvibe-modal-close" onclick="window.dispatchEvent(new Event('closeQuickVibeModal'))">✖️</span>
                """ + get_quickvibe_readme() + """
            </div>
            <script>
            window.addEventListener('closeQuickVibeModal', function() {
                window.parent.postMessage({isStreamlitMessage: true, type: 'streamlit:setComponentValue', key: 'show_examplevibes', value: false}, '*');
            });
            </script>
        """, unsafe_allow_html=True)
        # Add a close button workaround for Streamlit
        if st.button("Close exampleVibes"):
                    st.session_state['show_examplevibes'] = False
            
    with st.sidebar:
        # Add Groq badge at the top of sidebar
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
                    <img
                        src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg"
                        alt="Powered by Groq for fast inference."
                        style="width: 120px; height: auto;"
                    />
                </a>
            </div>
        """, unsafe_allow_html=True)
        
        # Add the QuickVibe logo to the sidebar
        st.image("static/logo.jpg", use_container_width=True)
        
        st.header("🔑 API Key Setup")
        st.markdown("Need a Groq API key to get this party started. Grab one from [GroqCloud](https://console.groq.com/keys)!")
        
        api_key_input = st.text_input(
            "Groq API Key", 
            type="password", 
            value=st.session_state.api_key,
            key="quickvibe_api_key_input",
            help="Paste your Groq API key here, legend."
        )
        
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            st.session_state.api_validated = False
            st.session_state.quickvibe_chat_models = []

        if st.button("🚀 Connect & Load Models", type="primary"):
            if st.session_state.api_key and st.session_state.api_key.strip():
                with st.spinner("Verifying key & fetching models... ✨"):
                    success, models_or_error = fetch_groq_models_for_quickvibe(st.session_state.api_key)
                    if success and isinstance(models_or_error, list):
                        st.session_state.quickvibe_chat_models = [m['id'] for m in models_or_error]
                        st.session_state.api_validated = True
                        st.session_state.current_model_index = 0
                        st.success(f"✅ Connected! Found {len(st.session_state.quickvibe_chat_models)} chat models. Let's gooo! 🔥")
                        st.balloons()
                    else:
                        st.session_state.api_validated = False
                        st.session_state.quickvibe_chat_models = []
                        st.error(f"❌ Connection Fail: {models_or_error}")
            else:
                st.error("Bruh, API key first. 🙄")
        
        if st.session_state.api_validated:
            st.success("🟢 API Live & Ready to Vibe!")
            if st.session_state.quickvibe_chat_models:
                st.markdown(f"**Models in rotation:** {len(st.session_state.quickvibe_chat_models)}")
        elif st.session_state.api_key:
            st.warning("🟡 API Key entered, but not validated. Smash that button!")
        else:
            st.info("🔴 No API Key. Add one to vibe.")
        
        # --- Groq Badge ---
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
                <img
                    src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg"
                    alt="Powered by Groq for fast inference."
                    style="height: 40px; opacity: 0.8;"
                />
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # --- ResponseVibe Toggle ---
        with st.expander("🎚️ ResponseVibe", expanded=True):
            vibe_options = [
                "Witty 😏",
                "Savage 🔥",
                "Supportive 🤗",
                "Dry 🧂",
                "Flirty 😘",
                "Chill 😎"
            ]
            
            current_vibe = st.session_state.get("quickvibe_response_vibe", "Witty 😏")
            try:
                current_index = vibe_options.index(current_vibe)
            except ValueError:
                current_index = 0
            
            vibe = st.radio(
                "Pick your vibe:",
                vibe_options,
                key="quickvibe_response_vibe_radio",
                index=current_index
            )
            
            # Only update if changed to prevent widget conflicts
            if vibe != st.session_state.get("quickvibe_response_vibe"):
                st.session_state["quickvibe_response_vibe"] = vibe

    if not _is_quickvibe_api_configured():
        st.warning("⚠️ Yo, hit up the sidebar & plug in your API key to start! 🔑")
        st.image("https://media.tenor.com/wpSo-8CrXqUAAAAi/loading-loading-forever.gif", caption="Waiting for that API key drip...")
        return

    for message in st.session_state.chat_history:
        avatar_icon = "😎" if message["role"] == "assistant" else "user"
        with st.chat_message(message["role"], avatar=avatar_icon if message["role"] == "assistant" else None):
            st.markdown(message["content"])
            if "model_used" in message:
                st.caption(f"🧠: {message['model_used']}")

    user_input = st.chat_input("What's the tea? 🍵 Spill the sitch or paste the text 👇", key="quickvibe_user_input")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        current_model = get_next_quickvibe_model()
        if not current_model:
            st.error("💀 No models loaded. Can't vibe. Check API key & models.")
            return

        with st.chat_message("assistant", avatar="😎"):
            with st.spinner(f"QuickVibe is cooking with {current_model}... 🍳"):
                success, response_content = send_quickvibe_message(current_model, user_input)
                if success:
                    st.markdown(response_content)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_content, "model_used": current_model})
                else:
                    st.error(response_content)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_content, "model_used": current_model, "error": True})
        st.rerun()

if __name__ == "__main__":
    quickvibe_main()

# Final check: Ensure all old function definitions and unused imports are removed.
# Removed: tempfile, json, base64, urlparse, dataclasses, functools, datetime, timezone, 
# audio_recorder_streamlit, asyncio, aiohttp, ClientTimeout as they are no longer directly used.
