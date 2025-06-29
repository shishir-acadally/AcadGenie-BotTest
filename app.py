import streamlit as st
import json
import traceback
import uuid
from typing import TypedDict, Annotated, Any
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from streamlit_oauth import OAuth2Component
import requests
from datetime import datetime
from pymongo import MongoClient

# set tracking for api/s
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "create_react_agent_test"

sys = """
You are AcadGenie, an expert academic assistant specializing in teaching through highly interactive, guided practice. Your role is to guide learners through complex concepts by breaking them down into small, manageable steps and using a variety of interactive prompts such as multiple-choice questions (MCQs), fill-in-the-blanks (FIBs), and short answers. You do not solve problems outright but help learners solve them through their own thinking, rooted in Socratic questioning, error analysis, and constructivist learning. You adjust the complexity of your responses based on the learner’s grade level and focus on fostering true conceptual clarity and real-world application.

Interaction Flow
Evaluate User Input:
	- If the user's input is not a question, engage in normal conversation: be helpful, supportive, and contextually appropriate.
	- If the user's input is a question, decompose it into sequential sub-problems or steps (e.g. identifying inputs and methods to use, recalling formulas, applying formulas, etc.).
	
Handling Complex Questions:
	- For each step, generate an interactive Multiple Choice Question(MCQ) to guide the learner:
	- Always include plausible options with distractor rationales (DRs) to address common misconceptions.
    - These question stem of MCQs can be of multiple forms like "Fill in the blank", "Short Answer", etc.
    - Vary the question type across steps to avoid repetition (e.g., don’t use same type of MCQs multiple times for similar steps).
    - Each step question should be in a flow that builds on the previous one, guiding the learner through the concept step by step. One such flow could be: identifying the inputs(if it is not given diretly), recalling the formula, applying the formula, and interpreting the results.
    - If multiple steps are similar to each-other, try combining them into one question.
    
Managing User Responses:
	If the learner answers correctly:
		- Confirm the step with specific feedback tied to conceptual clarity (e.g., “That’s right! You’ve correctly identified the formula.”).
		- Proceed to the next step with a new interactive prompt.
	If the learner answers incorrectly:
		On the first wrong attempt:
		- Identify the specific misconception using the format: "It seems you are confusing [concept A] with [concept B]."
		- Provide a hint or conceptual remedy and ask a follow-up question in a simpler way.
		On the second wrong attempt:
		- Reveal the correct answer with a detailed explanation, including why incorrect options (for MCQs) are wrong based on distractor rationales.
	Always guide the learner to think through each step; do not solve it for them.

Completing the Concept:
	- Once all steps are completed and understanding is confirmed, ask: "Would you like to explore another question?"
	- If yes, start again with a new question.

Tone and Style
	- Warm, supportive, and highly pedagogical, like a patient, chalk-and-talk tutor sitting beside the learner.
	- Use simple language, clear examples, and lots of reinforcement.
	- Foster true conceptual clarity and real-world application.
	- Avoid filler praise (e.g., “Great!”) unless tied to specific understanding.

Output Format
For interactive prompts, generate JSON objects with the following structure:
```json
{
  "conversation_message": "Your instructional or feedback message here.",
  "question_data": {
    "question": "Your question here?",
    "options": [
      {"option": "A", "text": "Option A text", "DR": "common misconception"},
      {"option": "B", "text": "Option B text", "DR": "common misconception"},
      {"option": "C", "text": "Option C text", "DR": "common misconception"},
      {"option": "D", "text": "Option D text", "DR": "common misconception"}
    ],
    "correct_answer": "correct option among all 4 options",
    "explanation": "Why the correct answer is correct and others are not.",
    "comment": "Encouraging or clarifying comment tied to understanding.",
  },
  "completed":"True/False according to whether the current question has been completely resolved."

}
```

For normal conversation, respond with:
``json
{
  "conversation_message": "Your normal conversation message here.",
  "question_data": {},
  "completed":"True/False according to whether the current conversation has been completed."
}
```

Important Notes
    - Always break down a question based on the learner's grade level.
	- Adjust the complexity of responses based on the learner's grade level.
	- Do not repeat the same type of question multiple times for similar steps.
	- Do not solve any step for the learner; always guide them to think through it.
	- Provide feedback that is specific and tied to conceptual clarity.
	- Never rush to solve the full problem; focus on one step at a time.
"""

# # Type definitions
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     remaining_steps: Any
#     memory: list

# def initialize_agent_state() -> State:
#     """Initialize a fresh agent state."""
#     return {
#         "messages": [],
#         "remaining_steps": None,
#         "memory": []
#     }

# def prompt(state: State, config: RunnableConfig) -> list[AnyMessage]:  
#     """Generate prompt with user context and memory."""
#     user_name = config["configurable"].get("user_name", "Student")
#     grade = config["configurable"].get("grade", "")
    
#     system_msg = f"{sys} Address the user as {user_name} from grade {grade}."
    
#     # Get memory from state
#     memory_messages = state.get("memory", [])
    
#     # Combine system message, memory, and current messages
#     all_messages = [SystemMessage(content=system_msg)] + memory_messages + state["messages"]
    
#     return all_messages

# def get_agent_response(user_input: str, config: dict) -> dict:
#     """Get response from the agent with proper memory management."""
#     try:
#         # Initialize agent memory in session state if not exists
#         if "agent_memory" not in st.session_state:
#             st.session_state.agent_memory = []
        
#         # Add user message to memory
#         user_message = HumanMessage(content=user_input, name="user")
#         st.session_state.agent_memory.append(user_message)
        
#         # Create agent if not exists
#         if "agent" not in st.session_state:
#             model = ChatOpenAI(
#                 model="gpt-4",  # Using gpt-4 instead of o3
#                 temperature=0.3,
#             api_key=st.secrets["OPENAI"]["OPENAI_API_KEY"]  # FIXED: Using secrets
#             )
            
#             checkpointer = InMemorySaver()
#             st.session_state.agent = create_react_agent(
#                 model=model,
#                 prompt=prompt,
#                 tools=[],
#                 checkpointer=checkpointer,
#             )
        
#         # Prepare state for agent
#         agent_state = {
#             "memory": st.session_state.agent_memory[:-1],  # Exclude the current message
#             "messages": [{"role": "user", "content": user_input}]
#         }
        
#         print(f"\n--- Asking Agent (User: {config['configurable'].get('user_name')}) ---")
#         print(f"Memory length: {len(st.session_state.agent_memory)}")
#         print(f"Memory: {st.session_state.agent_memory}")
#         # Invoke agent
#         response = st.session_state.agent.invoke(agent_state, config)
        
#         # Extract AI message from response
#         messages = response.get('messages', [])
#         ai_message = None
        
#         for msg in reversed(messages):
#             if isinstance(msg, AIMessage):
#                 ai_message = msg.content
#                 break
        
#         if not ai_message:
#             ai_message = "I'm sorry, I couldn't generate a response. Please try again."
        
#         # Add AI message to memory
#         st.session_state.agent_memory.append(AIMessage(content=ai_message, name="acadgenie"))
        
#         # Limit memory size (keep last 20 messages = 10 conversations)
#         if len(st.session_state.agent_memory) > 20:
#             st.session_state.agent_memory = st.session_state.agent_memory[-20:]
        
#         print(f"Agent responded. Memory now has {len(st.session_state.agent_memory)} messages")
        
#         return {
#             'human_message': user_input,
#             'acadgenie': ai_message,
#             'memory': st.session_state.agent_memory
#         }
        
#     except Exception as e:
#         print(f"Error in get_agent_response: {e}")
#         print(traceback.format_exc())
#         return {
#             'human_message': user_input,
#             'acadgenie': "I encountered an error while processing your request. Please try again.",
#             'memory': st.session_state.get('agent_memory', [])
#         }

# def parse_response(response_text: str) -> dict:
#     """Parse agent response - simplified version."""
#     if isinstance(response_text, dict):
#         return response_text
    
#     # Try to extract JSON if present
#     try:
#         first_bracket = response_text.find('{')
#         end_bracket = response_text.rfind('}')
#         if first_bracket != -1 and end_bracket != -1:
#             json_str = response_text[first_bracket:end_bracket + 1]
#             return json.loads(json_str)
#     except:
#         pass
    
#     # Return as simple conversation message
#     return {
#         "conversation_message": response_text
#     }

# # Google OAuth Configuration
# @st.cache_data
# def get_oauth_config():
#     """Get OAuth configuration from secrets."""
#     return {
#         "client_id": st.secrets["google"]["client_id"],
#         "client_secret": st.secrets["google"]["client_secret"],
#         "redirect_uri": "http://localhost:8501"
#     }

# def authenticate_user():
#     """Handle Google OAuth authentication."""
#     config = get_oauth_config()
    
#     oauth2 = OAuth2Component(
#         client_id=config["client_id"],
#         client_secret=config["client_secret"],
#         authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
#         token_endpoint="https://accounts.google.com/o/oauth2/token",
#         refresh_token_endpoint="https://accounts.google.com/o/oauth2/token",
#         revoke_token_endpoint="https://accounts.google.com/o/oauth2/revoke",
#     )
    
#     scope = "openid email profile"
    
#     result = oauth2.authorize_button(
#         name="Continue with Google",
#         icon="https://developers.google.com/identity/images/g-logo.png",
#         redirect_uri=config["redirect_uri"],
#         scope=scope,
#         key="google_oauth",
#         use_container_width=True
#     )
    
#     return result

# def get_user_info(access_token: str) -> dict:
#     """Fetch user information from Google using access token."""
#     try:
#         response = requests.get(
#             'https://www.googleapis.com/oauth2/v2/userinfo',
#             headers={'Authorization': f'Bearer {access_token}'},
#             timeout=10
#         )
        
#         if response.status_code == 200:
#             return response.json()
#         else:
#             st.error(f"Failed to get user info: {response.status_code}")
#             return None
#     except Exception as e:
#         st.error(f"Error fetching user info: {e}")
#         return None

# def clear_user_session():
#     """Clear all user session data."""
#     keys_to_clear = list(st.session_state.keys())
#     for key in keys_to_clear:
#         del st.session_state[key]

# def initialize_session_state():
#     """Initialize session state variables."""
#     defaults = {
#         "chat_history": [],
#         "user_email": "",
#         "user_name": "",
#         "user_grade": "",
#         "authenticated": False,
#         "setup_complete": False,
#         "thread_id": str(uuid.uuid4()),
#         "agent_memory": []
#     }
    
#     for key, default_value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = default_value

# def render_authentication():
#     """Render the authentication page."""
#     st.title("🤖 AcadGenie")
#     st.markdown("Your AI learning companion. Please sign in to continue.")
    
#     auth_result = authenticate_user()
    
#     if auth_result and 'token' in auth_result:
#         token_data = auth_result['token']
        
#         # Extract access token
#         access_token = None
#         if isinstance(token_data, dict) and 'access_token' in token_data:
#             access_token = token_data['access_token']
#         elif isinstance(token_data, str):
#             access_token = token_data
        
#         if access_token:
#             user_info = get_user_info(access_token)
            
#             if user_info:
#                 st.session_state.user_email = user_info.get('email', '')
#                 st.session_state.user_name = user_info.get('name', '')
#                 st.session_state.authenticated = True
#                 st.success(f"Successfully logged in as {st.session_state.user_name}")
#                 st.rerun()
#             else:
#                 st.error("Failed to get user information")
#         else:
#             st.error("Authentication failed - no access token received")

# def render_grade_setup():
#     """Render the grade selection page."""
#     st.title("🤖 AcadGenie")
#     st.markdown(f"Welcome, {st.session_state.user_name}!")
#     st.markdown(f"Email: {st.session_state.user_email}")
    
#     st.markdown("### Please select your grade level")
    
#     with st.form("grade_setup"):
#         grade = st.selectbox(
#             "Your Grade:",
#             options=["", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
#             index=0
#         )
        
#         submitted = st.form_submit_button("Start Learning")
        
#         if submitted:
#             if grade:
#                 st.session_state.user_grade = grade
#                 st.session_state.setup_complete = True
#                 st.rerun()
#             else:
#                 st.error("Please select your grade to continue.")

# def render_chat_interface():
#     """Render the main chat interface."""
#     st.title("🤖 AcadGenie")
    
#     # Header with user info and controls
#     col1, col2, col3 = st.columns([2, 1, 1])
#     with col1:
#         st.markdown(f"**{st.session_state.user_name}** | Grade {st.session_state.user_grade}")
#     with col2:
#         if st.button("Clear Chat", type="secondary"):
#             st.session_state.chat_history = []
#             st.session_state.agent_memory = []
#             st.rerun()
#     with col3:
#         if st.button("Logout", type="secondary"):
#             clear_user_session()
#             st.rerun()
    
#     st.markdown("---")
    
#     # Prepare config for agent
#     config = {
#         "configurable": {
#             "user_name": st.session_state.user_name,
#             "grade": st.session_state.user_grade,
#             "thread_id": st.session_state.thread_id
#         }
#     }
    
#     # Chat input
#     user_input = st.chat_input("Type your message here...")
    
#     if user_input:
#         # Add user message to display history
#         st.session_state.chat_history.append(("You", user_input))
        
#         with st.spinner("AcadGenie is thinking..."):
#             try:
#                 # Get response from agent
#                 result = get_agent_response(user_input, config)
                
#                 # Parse and format the response
#                 ai_response = result.get("acadgenie", "I'm not sure how to respond!")
#                 parsed_response = parse_response(ai_response)
                
#                 # Format message for display
#                 formatted_msg = format_ai_message(parsed_response)
                
#                 # Add to display history
#                 st.session_state.chat_history.append(("AcadGenie", formatted_msg))
                
#             except Exception as e:
#                 error_msg = "I encountered an error while processing your request. Please try again."
#                 st.session_state.chat_history.append(("AcadGenie", error_msg))
#                 st.error(f"Error: {str(e)}")
    
#     # Display chat history
#     for sender, message in st.session_state.chat_history:
#         with st.chat_message("user" if sender == "You" else "assistant"):
#             st.markdown(message)

# def format_ai_message(parsed_response: dict) -> str:
#     """Format AI response for display."""
#     if isinstance(parsed_response, dict):
#         formatted_msg = parsed_response.get("conversation_message", "")
        
#         # Handle structured question data if present
#         question_data = parsed_response.get("question_data")
#         if question_data:
#             formatted_msg += f"\n\n**Question:** {question_data.get('question', '')}\n"
            
#             if question_data.get('type') == 'MCQ':
#                 formatted_msg += "\n**Options:**\n"
#                 for option in question_data.get('options', []):
#                     option_letter = option.get('option', '')
#                     option_text = option.get('text', '')
#                     formatted_msg += f"- {option_letter}: {option_text}\n"
            
#             formatted_msg += f"\n**Correct Answer:** {question_data.get('correct_answer', '')}"
#             formatted_msg += f"\n**Explanation:** {question_data.get('explanation', '')}"
            
#             comment = question_data.get('comment')
#             if comment:
#                 formatted_msg += f"\n**Comment:** {comment}"
        
#         return formatted_msg if formatted_msg else str(parsed_response)
#     else:
#         return str(parsed_response)

# def main():
#     """Main application function."""
#     # Page configuration
#     st.set_page_config(
#         page_title="AcadGenie Chat",
#         page_icon="🧠",
#         layout="centered"
#     )
    
#     # Initialize session state
#     initialize_session_state()
    
#     # Route to appropriate page based on authentication state
#     if not st.session_state.authenticated:
#         render_authentication()
#     elif not st.session_state.setup_complete:
#         render_grade_setup()
#     else:
#         render_chat_interface()

# if __name__ == "__main__":
#     main()

# # Type definitions
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     remaining_steps: Any
#     memory: list

# def initialize_agent_state() -> State:
#     """Initialize a fresh agent state."""
#     return {
#         "messages": [],
#         "remaining_steps": None,
#         "memory": []
#     }

# def prompt(state: State, config: RunnableConfig) -> list[AnyMessage]:  
#     """Generate prompt with user context and memory."""
#     user_name = config["configurable"].get("user_name", "Student")
#     grade = config["configurable"].get("grade", "")
    
#     system_msg = f"{sys} Address the user as {user_name} from grade {grade}."
    
#     # Get memory from state
#     memory_messages = state.get("memory", [])
    
#     # Combine system message, memory, and current messages
#     all_messages = [SystemMessage(content=system_msg)] + memory_messages + state["messages"]
    
#     return all_messages

# def get_agent_response(user_input: str, config: dict) -> dict:
#     """Get response from the agent with proper memory management."""
#     try:
#         # Initialize agent memory in session state if not exists
#         if "agent_memory" not in st.session_state:
#             st.session_state.agent_memory = []
        
#         # Add user message to memory
#         user_message = HumanMessage(content=user_input, name="user")
#         st.session_state.agent_memory.append(user_message)
        
#         # Create agent if not exists
#         if "agent" not in st.session_state:
#             model = ChatOpenAI(
#                 model="gpt-4",  # Using gpt-4 instead of o3
#                 temperature=0.3,
#             api_key=st.secrets["OPENAI"]["OPENAI_API_KEY"]  # FIXED: Using secrets
#             )
            
#             checkpointer = InMemorySaver()
#             st.session_state.agent = create_react_agent(
#                 model=model,
#                 prompt=prompt,
#                 tools=[],
#                 checkpointer=checkpointer,
#             )
        
#         # Prepare state for agent
#         agent_state = {
#             "memory": st.session_state.agent_memory[:-1],  # Exclude the current message
#             "messages": [{"role": "user", "content": user_input}]
#         }
        
#         print(f"\n--- Asking Agent (User: {config['configurable'].get('user_name')}) ---")
#         print(f"Memory length: {len(st.session_state.agent_memory)}")
#         print(f"Memory: {st.session_state.agent_memory}")
#         # Invoke agent
#         response = st.session_state.agent.invoke(agent_state, config)
        
#         # Extract AI message from response
#         messages = response.get('messages', [])
#         ai_message = None
        
#         for msg in reversed(messages):
#             if isinstance(msg, AIMessage):
#                 ai_message = msg.content
#                 break
        
#         if not ai_message:
#             ai_message = "I'm sorry, I couldn't generate a response. Please try again."
        
#         # Add AI message to memory
#         st.session_state.agent_memory.append(AIMessage(content=ai_message, name="acadgenie"))
        
#         # Limit memory size (keep last 20 messages = 10 conversations)
#         if len(st.session_state.agent_memory) > 20:
#             st.session_state.agent_memory = st.session_state.agent_memory[-20:]
        
#         print(f"Agent responded. Memory now has {len(st.session_state.agent_memory)} messages: {st.session_state.agent_memory}")
        
#         return {
#             'human_message': user_input,
#             'acadgenie': ai_message,
#             'memory': st.session_state.agent_memory
#         }
        
#     except Exception as e:
#         print(f"Error in get_agent_response: {e}")
#         print(traceback.format_exc())
#         return {
#             'human_message': user_input,
#             'acadgenie': "I encountered an error while processing your request. Please try again.",
#             'memory': st.session_state.get('agent_memory', [])
#         }

# def parse_response(response_text: str) -> dict:
#     """Parse agent response - simplified version."""
#     if isinstance(response_text, dict):
#         return response_text
    
#     # Try to extract JSON if present
#     try:
#         first_bracket = response_text.find('{')
#         end_bracket = response_text.rfind('}')
#         if first_bracket != -1 and end_bracket != -1:
#             json_str = response_text[first_bracket:end_bracket + 1]
#             return json.loads(json_str)
#     except:
#         pass
    
#     # Return as simple conversation message
#     return {
#         "conversation_message": response_text
#     }

# # Google OAuth Configuration
# @st.cache_data
# def get_oauth_config():
#     """Get OAuth configuration from secrets."""
#     return {
#         "client_id": st.secrets["google"]["client_id"],
#         "client_secret": st.secrets["google"]["client_secret"],
#         "redirect_uri": "http://localhost:8501"
#     }

# def authenticate_user():
#     """Handle Google OAuth authentication."""
#     config = get_oauth_config()
    
#     oauth2 = OAuth2Component(
#         client_id=config["client_id"],
#         client_secret=config["client_secret"],
#         authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
#         token_endpoint="https://accounts.google.com/o/oauth2/token",
#         refresh_token_endpoint="https://accounts.google.com/o/oauth2/token",
#         revoke_token_endpoint="https://accounts.google.com/o/oauth2/revoke",
#     )
    
#     scope = "openid email profile"
    
#     result = oauth2.authorize_button(
#         name="Continue with Google",
#         icon="https://developers.google.com/identity/images/g-logo.png",
#         redirect_uri=config["redirect_uri"],
#         scope=scope,
#         key="google_oauth",
#         use_container_width=True
#     )
    
#     return result

# def get_user_info(access_token: str) -> dict:
#     """Fetch user information from Google using access token."""
#     try:
#         response = requests.get(
#             'https://www.googleapis.com/oauth2/v2/userinfo',
#             headers={'Authorization': f'Bearer {access_token}'},
#             timeout=10
#         )
        
#         if response.status_code == 200:
#             return response.json()
#         else:
#             st.error(f"Failed to get user info: {response.status_code}")
#             return None
#     except Exception as e:
#         st.error(f"Error fetching user info: {e}")
#         return None

# def clear_user_session():
#     """Clear all user session data."""
#     keys_to_clear = list(st.session_state.keys())
#     for key in keys_to_clear:
#         del st.session_state[key]

# def initialize_session_state():
#     """Initialize session state variables."""
#     defaults = {
#         "chat_history": [],
#         "user_email": "",
#         "user_name": "",
#         "user_grade": "",
#         "authenticated": False,
#         "setup_complete": False,
#         "thread_id": str(uuid.uuid4()),
#         "agent_memory": [],
#         "feedback_data": [],  # New: Store feedback data with memory
#         "show_feedback_popup": False,  # New: Control popup visibility
#         "current_feedback_index": -1,  # New: Track which message is being rated
#         "pending_feedback": False,  # New: Track if feedback is pending
#         "feedback_locked": [],  # New: Track which messages have feedback locked
#         "temp_feedback": {"type": None, "reason": ""},  # New: Store temporary feedback
#     }
    
#     for key, default_value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = default_value

# def save_feedback_session():
#     """Save feedback session data to session state with new structure."""
#     # Create a deep copy of agent memory for this feedback session
#     memory_copy = []
    
#     for msg in st.session_state.agent_memory:
#         if hasattr(msg, 'content') and hasattr(msg, 'name'):
#             msg_copy = {
#                 "content": msg.content,
#                 "name": msg.name,
#                 "type": "HumanMessage" if msg.name == "user" else "AIMessage"
#             }
            
#             # Add feedback data to AI messages
#             if msg.name == "acadgenie":
#                 msg_copy["feedback_type"] = st.session_state.temp_feedback["type"]
#                 msg_copy["feedback_reason"] = st.session_state.temp_feedback["reason"]
            
#             memory_copy.append(msg_copy)
    
#     # Clear existing feedback_data and create new structure with exactly 2 objects
#     st.session_state.feedback_data = []
    
#     # Index 0: Session info object
#     session_info = {
#         "user_email": st.session_state.user_email,
#         "user_name": st.session_state.user_name,
#         "user_grade": st.session_state.user_grade,
#         "timestamp": datetime.now().isoformat(),
#         "session_id": st.session_state.thread_id
#     }
    
#     # Index 1: Memory object
#     memory_object = {
#         "memory": memory_copy
#     }
    
#     st.session_state.feedback_data.append(session_info)
#     st.session_state.feedback_data.append(memory_object)
    
#     # Reset temporary feedback
#     st.session_state.temp_feedback = {"type": None, "reason": ""}
#     st.session_state.pending_feedback = False
    
#     print(f"Feedback session saved: {len(memory_copy)} messages")
#     print(f"Feedback data: {st.session_state.feedback_data}")

# def set_temp_feedback(feedback_type: str, reason: str = ""):
#     """Set temporary feedback before saving."""
#     st.session_state.temp_feedback = {
#         "type": feedback_type,
#         "reason": reason
#     }

# def render_feedback_popup(message_index: int):
#     """Render the feedback popup for thumbs down positioned below the buttons."""
#     # Create a container for the popup that appears right after the buttons
#     with st.container():
#         # Add some visual styling for the popup
#         st.markdown("""
#         <div style="
#             background-color: #f0f2f6;
#             padding: 15px;
#             border-radius: 10px;
#             border-left: 4px solid #ff6b6b;
#             margin: 10px 0;
#         ">
#         """, unsafe_allow_html=True)
        
#         st.markdown("### 👎 Help us improve")
#         st.markdown("Please tell us what went wrong:")
        
#         # Feedback form
#         with st.form(f"feedback_form_{message_index}", clear_on_submit=False):
#             reason = st.text_area(
#                 "Reason for thumbs down:",
#                 placeholder="e.g., Incorrect information, Not helpful, Confusing explanation...",
#                 height=100,
#                 value=st.session_state.temp_feedback.get("reason", ""),
#                 key=f"feedback_reason_{message_index}"
#             )
            
#             col_submit, col_cancel = st.columns(2)
            
#             with col_submit:
#                 submitted = st.form_submit_button("Submit Feedback", type="primary")
            
#             with col_cancel:
#                 cancelled = st.form_submit_button("Cancel")
            
#             if submitted:
#                 set_temp_feedback("thumbs_down", reason)
#                 st.session_state.show_feedback_popup = False
#                 st.session_state.current_feedback_index = -1
#                 st.success("Feedback recorded! Click 'Save' to finalize.")
#                 st.rerun()
            
#             if cancelled:
#                 st.session_state.show_feedback_popup = False
#                 st.session_state.current_feedback_index = -1
#                 st.rerun()
        
#         st.markdown("</div>", unsafe_allow_html=True)

# def render_feedback_buttons(message_index: int):
#     """Render thumbs up, thumbs down, and save buttons for a message."""
#     # Check if this message feedback is locked
#     is_locked = message_index in st.session_state.feedback_locked
    
#     if is_locked:
#         # Show locked feedback state
#         col1, col2 = st.columns([10, 2])
#         with col2:
#             # Find the feedback for this message from feedback_data
#             feedback_type = None
#             if st.session_state.feedback_data and len(st.session_state.feedback_data) >= 2:
#                 memory_object = st.session_state.feedback_data[1]
#                 memory = memory_object.get("memory", [])
#                 ai_messages = [msg for msg in memory if msg.get("type") == "AIMessage"]
#                 if ai_messages:
#                     latest_ai = ai_messages[-1]
#                     feedback_type = latest_ai.get("feedback_type")
            
#             if feedback_type == "thumbs_up":
#                 st.markdown("✅ 👍")
#             elif feedback_type == "thumbs_down":
#                 st.markdown("✅ 👎")
#     else:
#         # Show interactive feedback buttons with styling
#         current_selection = st.session_state.temp_feedback["type"]
        
#         # Simple CSS for button styling
#         st.markdown("""
#         <style>
#         div[data-testid="stButton"] > button {
#             transition: all 0.2s ease;
#         }
#         </style>
#         """, unsafe_allow_html=True)
        
#         col1, col2, col3, col4 = st.columns([8, 1, 1, 2])
        
#         # Determine if feedback has been selected
#         feedback_selected = current_selection is not None
        
#         with col2:
#             # Thumbs up button
#             thumbs_up_pressed = st.button(
#                 "👍", 
#                 key=f"thumbs_up_{message_index}", 
#                 help="This response was helpful"
#             )
#             if thumbs_up_pressed:
#                 set_temp_feedback("thumbs_up")
#                 # Close any open popup
#                 st.session_state.show_feedback_popup = False
#                 st.session_state.current_feedback_index = -1
#                 st.rerun()
        
#         with col3:
#             # Thumbs down button
#             thumbs_down_pressed = st.button(
#                 "👎", 
#                 key=f"thumbs_down_{message_index}", 
#                 help="This response needs improvement"
#             )
#             if thumbs_down_pressed:
#                 # Toggle popup for this specific message
#                 if (st.session_state.show_feedback_popup and 
#                     st.session_state.current_feedback_index == message_index):
#                     # Close popup if it's already open for this message
#                     st.session_state.show_feedback_popup = False
#                     st.session_state.current_feedback_index = -1
#                 else:
#                     # Open popup for this message
#                     st.session_state.show_feedback_popup = True
#                     st.session_state.current_feedback_index = message_index
#                     set_temp_feedback("thumbs_down", "")  # Reset reason for new selection
#                 st.rerun()
        
#         with col4:
#             save_pressed = st.button(
#                 "💾 Save", 
#                 key=f"save_feedback_{message_index}",
#                 disabled=not feedback_selected,
#                 type="primary" if feedback_selected else "secondary"
#             )
#             if save_pressed and feedback_selected:
#                 save_feedback_session()
#                 st.session_state.feedback_locked.append(message_index)
#                 # Close popup after saving
#                 st.session_state.show_feedback_popup = False
#                 st.session_state.current_feedback_index = -1
#                 st.success("Feedback saved successfully!")
#                 st.rerun()
        
#         # Show current selection with visual indicators
#         if feedback_selected:
#             with col1:
#                 if current_selection == "thumbs_up":
#                     st.markdown("**🔵 Selected:** 👍 Helpful")
#                 elif current_selection == "thumbs_down":
#                     reason = st.session_state.temp_feedback.get("reason", "")
#                     if reason:
#                         st.markdown(f"**🔵 Selected:** 👎 Needs improvement - {reason[:50]}...")
#                     else:
#                         st.markdown("**🔵 Selected:** 👎 Needs improvement (click 👎 again to add reason)")
    
#     # Render the feedback popup RIGHT HERE if it should be shown for this message
#     if (st.session_state.show_feedback_popup and 
#         st.session_state.current_feedback_index == message_index):
#         render_feedback_popup(message_index)

# def save_feedback(message_index: int, feedback_type: str, reason: str = ""):
#     """Save feedback data to session state."""
#     feedback_entry = {
#         "timestamp": datetime.now().isoformat(),
#         "message_index": message_index,
#         "user_message": st.session_state.chat_history[message_index-1][1] if message_index > 0 else "",
#         "ai_message": st.session_state.chat_history[message_index][1],
#         "feedback_type": feedback_type,  # "thumbs_up" or "thumbs_down"
#         "reason": reason,
#         "user_email": st.session_state.user_email,
#         "user_name": st.session_state.user_name,
#         "user_grade": st.session_state.user_grade
#     }
    
#     st.session_state.feedback_data.append(feedback_entry)
#     print(f"Feedback saved: {feedback_entry}")

# def render_authentication():
#     """Render the authentication page."""
#     st.title("🤖 AcadGenie")
#     st.markdown("Your AI learning companion. Please sign in to continue.")
    
#     auth_result = authenticate_user()
    
#     if auth_result and 'token' in auth_result:
#         token_data = auth_result['token']
        
#         # Extract access token
#         access_token = None
#         if isinstance(token_data, dict) and 'access_token' in token_data:
#             access_token = token_data['access_token']
#         elif isinstance(token_data, str):
#             access_token = token_data
        
#         if access_token:
#             user_info = get_user_info(access_token)
            
#             if user_info:
#                 st.session_state.user_email = user_info.get('email', '')
#                 st.session_state.user_name = user_info.get('name', '')
#                 st.session_state.authenticated = True
#                 st.success(f"Successfully logged in as {st.session_state.user_name}")
#                 st.rerun()
#             else:
#                 st.error("Failed to get user information")
#         else:
#             st.error("Authentication failed - no access token received")

# def render_grade_setup():
#     """Render the grade selection page."""
#     st.title("🤖 AcadGenie")
#     st.markdown(f"Welcome, {st.session_state.user_name}!")
#     st.markdown(f"Email: {st.session_state.user_email}")
    
#     st.markdown("### Please select your grade level")
    
#     with st.form("grade_setup"):
#         grade = st.selectbox(
#             "Your Grade:",
#             options=["", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
#             index=0
#         )
        
#         submitted = st.form_submit_button("Start Learning")
        
#         if submitted:
#             if grade:
#                 st.session_state.user_grade = grade
#                 st.session_state.setup_complete = True
#                 st.rerun()
#             else:
#                 st.error("Please select your grade to continue.")

# def render_chat_interface():
#     """Render the main chat interface."""
#     st.title("🤖 AcadGenie")
    
#     # Header with user info and controls
#     col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
#     with col1:
#         st.markdown(f"**{st.session_state.user_name}** | Grade {st.session_state.user_grade}")
#     with col2:
#         if st.button("View Feedback", type="secondary"):
#             st.session_state.show_feedback_summary = not st.session_state.get('show_feedback_summary', False)
#     with col3:
#         if st.button("Clear Chat", type="secondary"):
#             st.session_state.chat_history = []
#             st.session_state.agent_memory = []
#             st.session_state.feedback_locked = []
#             st.session_state.pending_feedback = False
#             st.session_state.temp_feedback = {"type": None, "reason": ""}
#             st.rerun()
#     with col4:
#         if st.button("Logout", type="secondary"):
#             clear_user_session()
#             st.rerun()
    
#     # Show feedback summary if requested
#     if st.session_state.get('show_feedback_summary', False):
#         with st.expander("📊 Feedback Summary", expanded=True):
#             if st.session_state.feedback_data and len(st.session_state.feedback_data) >= 2:
#                 # Get session info from index 0
#                 session_info = st.session_state.feedback_data[0]
#                 # Get memory from index 1
#                 memory_object = st.session_state.feedback_data[1]
#                 memory = memory_object.get("memory", [])
                
#                 st.write(f"**Session ID:** {session_info.get('session_id', 'N/A')}")
#                 st.write(f"**User:** {session_info.get('user_name', 'N/A')} (Grade {session_info.get('user_grade', 'N/A')})")
#                 st.write(f"**Timestamp:** {session_info.get('timestamp', 'N/A')}")
                
#                 # Count feedback types from memory
#                 thumbs_up_count = 0
#                 thumbs_down_count = 0
#                 recent_improvements = []
                
#                 for msg in memory:
#                     if msg.get("type") == "AIMessage" and "feedback_type" in msg:
#                         if msg["feedback_type"] == "thumbs_up":
#                             thumbs_up_count += 1
#                         elif msg["feedback_type"] == "thumbs_down":
#                             thumbs_down_count += 1
#                             if msg.get("feedback_reason"):
#                                 recent_improvements.append(msg["feedback_reason"])
                
#                 col_up, col_down = st.columns(2)
#                 with col_up:
#                     st.metric("👍 Positive", thumbs_up_count)
#                 with col_down:
#                     st.metric("👎 Needs Improvement", thumbs_down_count)
                
#                 if recent_improvements:
#                     st.markdown("**Recent improvement suggestions:**")
#                     for reason in recent_improvements[-3:]:  # Show last 3
#                         st.markdown(f"- {reason}")
                        
#                 st.write(f"**Total messages in memory:** {len(memory)}")
#             else:
#                 st.info("No feedback data available yet.")
    
#     st.markdown("---")
    
#     # Render feedback popup if needed - MOVE TO TOP
#     if st.session_state.show_feedback_popup:
#         render_feedback_popup()
    
#     # Prepare config for agent
#     config = {
#         "configurable": {
#             "user_name": st.session_state.user_name,
#             "grade": st.session_state.user_grade,
#             "thread_id": st.session_state.thread_id
#         }
#     }
    
#     # Chat input - only allow if no pending feedback
#     if not st.session_state.pending_feedback:
#         user_input = st.chat_input("Type your message here...")
#     else:
#         st.chat_input("Please provide feedback on the previous response before continuing...", disabled=True)
#         user_input = None
    
#     if user_input:
#         # Add user message to display history
#         st.session_state.chat_history.append(("You", user_input))
        
#         with st.spinner("AcadGenie is thinking..."):
#             try:
#                 # Get response from agent
#                 result = get_agent_response(user_input, config)
                
#                 # Parse and format the response
#                 ai_response = result.get("acadgenie", "I'm not sure how to respond!")
#                 parsed_response = parse_response(ai_response)
#                 print("Agent Response: ", parsed_response)
#                 # Format message for display
#                 formatted_msg = format_ai_message(parsed_response)
                
#                 # Add to display history
#                 st.session_state.chat_history.append(("AcadGenie", formatted_msg))
                
#                 # Set pending feedback for the new AI message
#                 st.session_state.pending_feedback = True
                
#             except Exception as e:
#                 error_msg = "I encountered an error while processing your request. Please try again."
#                 st.session_state.chat_history.append(("AcadGenie", error_msg))
#                 st.session_state.pending_feedback = True  # Also require feedback for error messages
#                 st.error(f"Error: {str(e)}")
    
#     # Display chat history with feedback buttons
#     for i, (sender, message) in enumerate(st.session_state.chat_history):
#         with st.chat_message("user" if sender == "You" else "assistant"):
#             st.markdown(message)
            
#             # Add feedback buttons only for AcadGenie messages
#             if sender == "AcadGenie":
#                 render_feedback_buttons(i)
    
#     # Show pending feedback warning if needed
#     if st.session_state.pending_feedback:
#         st.warning("⚠️ Please provide feedback on the above response before sending another message.")

# def format_ai_message(parsed_response: dict) -> str:
#     """Format AI response for display."""
#     if isinstance(parsed_response, dict):
#         formatted_msg = parsed_response.get("conversation_message", "")
        
#         # Handle structured question data if present
#         question_data = parsed_response.get("question_data")
#         if question_data:
#             formatted_msg += f"\n\n**Question:** {question_data.get('question', '')}\n"
            
#             if question_data.get('type') == 'MCQ':
#                 formatted_msg += "\n**Options:**\n"
#                 for option in question_data.get('options', []):
#                     option_letter = option.get('option', '')
#                     option_text = option.get('text', '')
#                     formatted_msg += f"- {option_letter}: {option_text}\n"
            
#             formatted_msg += f"\n**Correct Answer:** {question_data.get('correct_answer', '')}"
#             formatted_msg += f"\n**Explanation:** {question_data.get('explanation', '')}"
            
#             comment = question_data.get('comment')
#             if comment:
#                 formatted_msg += f"\n**Comment:** {comment}"
        
#         return formatted_msg if formatted_msg else str(parsed_response)
#     else:
#         return str(parsed_response)

# def main():
#     """Main application function."""
#     # Page configuration
#     st.set_page_config(
#         page_title="AcadGenie Chat",
#         page_icon="🧠",
#         layout="centered"
#     )
    
#     # Initialize session state
#     initialize_session_state()
    
#     # Route to appropriate page based on authentication state
#     if not st.session_state.authenticated:
#         render_authentication()
#     elif not st.session_state.setup_complete:
#         render_grade_setup()
#     else:
#         render_chat_interface()

# if __name__ == "__main__":
#     main()

# Define the add_messages function for TypedDict annotation
def add_messages(left: list, right: list) -> list:
    """Merge two lists of messages."""
    return left + right

class State(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: Any
    memory: list

def initialize_agent_state() -> State:
    """Initialize a fresh agent state."""
    return {
        "messages": [],
        "remaining_steps": None,
        "memory": []
    }

def push_feedback_data():
    """Push feedback data to database."""
    try:
        client = MongoClient(st.secrets["mongo"]["db_uri"])
        db = client[st.secrets["mongo"]["db_name"]]
        collection = db[st.secrets["mongo"]["collection_name"]]
        feedback = {'session_info':st.session_state.feedback_data[0], 'history':st.session_state.feedback_data[1]['memory']}
        inserted_id = collection.insert_one(feedback).inserted_id
        print(f"Feedback data pushed to database with ID: {inserted_id}")
        return True 
    except Exception as e:
        print(f"Error pushing feedback data: {e}")
        return False


def prompt(state: State, config: RunnableConfig) -> list[AnyMessage]:  
    """Generate prompt with user context and memory."""
    user_name = config["configurable"].get("user_name", "Student")
    grade = config["configurable"].get("grade", "")
    
    system_msg = f"{sys} Address the user as {user_name} from grade {grade}."
    
    # Get memory from state
    memory_messages = state.get("memory", [])
    
    # Combine system message, memory, and current messages
    all_messages = [SystemMessage(content=system_msg)] + memory_messages + state["messages"]
    
    return all_messages

def initialize_feedback_data():
    """Initialize feedback data structure only if it doesn't exist or is incomplete."""
    if ("feedback_data" not in st.session_state or 
        not st.session_state.feedback_data or 
        len(st.session_state.feedback_data) < 2):
        
        session_info = {
            "user_email": st.session_state.user_email,
            "user_name": st.session_state.user_name,
            "user_grade": st.session_state.user_grade,
            "timestamp": datetime.now().isoformat(),
            "session_id": st.session_state.thread_id
        }
        
        st.session_state.feedback_data = [session_info, {"memory": []}]

def get_agent_response(user_input: str, config: dict) -> dict:
    """Get response from the agent with proper memory management."""
    try:
        # Initialize agent memory in session state if not exists
        if "agent_memory" not in st.session_state:
            st.session_state.agent_memory = []
        
        # Add user message to memory
        user_message = HumanMessage(content=user_input, name="user")
        st.session_state.agent_memory.append(user_message)
        
        # Add user message to feedback_data immediately
        st.session_state.feedback_data[1]['memory'].append({
            "content": user_input,
            "name": 'user',
            "type": "HumanMessage"
        })
        
        # Create agent if not exists
        if "agent" not in st.session_state:
            model = ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
                api_key=st.secrets["OPENAI"]["OPENAI_API_KEY"]
            )
            checkpointer = InMemorySaver()
            st.session_state.agent = create_react_agent(
                model=model,
                prompt=prompt,
                tools=[],
                checkpointer=checkpointer,
            )
        
        # Prepare state for agent
        agent_state = {
            "memory": st.session_state.agent_memory[:-1],  # Exclude the current message
            "messages": [user_message]  # Use the actual message object
        }
        
        print(f"\n--- Asking Agent (User: {config['configurable'].get('user_name')}) ---")
        print(f"Memory length: {len(st.session_state.agent_memory)}")
        
        # Invoke agent with thread_id for checkpointing
        thread_id = config["configurable"].get("thread_id", str(uuid.uuid4()))
        response = st.session_state.agent.invoke(
            agent_state, 
            {"configurable": {**config["configurable"], "thread_id": thread_id}}
        )
        
        # Extract AI message from response
        messages = response.get('messages', [])
        ai_message = None
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break
        
        if not ai_message:
            ai_message = "I'm sorry, I couldn't generate a response. Please try again."
        
        # Add AI message to memory
        ai_message_obj = AIMessage(content=ai_message, name="acadgenie")
        st.session_state.agent_memory.append(ai_message_obj)
        
        # Add AI message to feedback_data immediately (without feedback initially)
        st.session_state.feedback_data[1]['memory'].append({
            "content": ai_message,
            "name": 'acadgenie',
            "type": "AIMessage",
            "feedback_type": None,  # Will be updated when user provides feedback
            "feedback_reason": None
        })
        
        # Limit memory size (keep last 20 messages = 10 conversations)
        if len(st.session_state.agent_memory) > 20:
            st.session_state.agent_memory = st.session_state.agent_memory[-20:]
            # Also limit feedback_data memory
            if len(st.session_state.feedback_data[1]['memory']) > 20:
                st.session_state.feedback_data[1]['memory'] = st.session_state.feedback_data[1]['memory'][-20:]
        
        print(f"Agent responded. Memory now has {len(st.session_state.agent_memory)} messages")
        print(f"Feedback data memory now has {len(st.session_state.feedback_data[1]['memory'])} messages")
        
        return {
            'human_message': user_input,
            'acadgenie': ai_message,
            'memory': st.session_state.agent_memory
        }
        
    except Exception as e:
        print(f"Error in get_agent_response: {e}")
        print(traceback.format_exc())
        return {
            'human_message': user_input,
            'acadgenie': "I encountered an error while processing your request. Please try again.",
            'memory': st.session_state.get('agent_memory', [])
        }

def parse_response(response_text: str) -> dict:
    """Parse agent response - simplified version."""
    if isinstance(response_text, dict):
        return response_text
    
    # Try to extract JSON if present
    try:
        first_bracket = response_text.find('{')
        end_bracket = response_text.rfind('}')
        if first_bracket != -1 and end_bracket != -1:
            json_str = response_text[first_bracket:end_bracket + 1]
            return json.loads(json_str)
    except:
        pass
    
    # Return as simple conversation message
    return {
        "conversation_message": response_text
    }

# Google OAuth Configuration
@st.cache_data
def get_oauth_config():
    """Get OAuth configuration from secrets."""
    return {
        "client_id": st.secrets["google"]["client_id"],
        "client_secret": st.secrets["google"]["client_secret"],
        "redirect_uri": st.secrets["google"]["redirect_url"]
    }

def authenticate_user():
    """Handle Google OAuth authentication."""
    config = get_oauth_config()
    
    oauth2 = OAuth2Component(
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://accounts.google.com/o/oauth2/token",
        refresh_token_endpoint="https://accounts.google.com/o/oauth2/token",
        revoke_token_endpoint="https://accounts.google.com/o/oauth2/revoke",
    )
    
    scope = "openid email profile"
    
    result = oauth2.authorize_button(
        name="Continue with Google",
        icon="https://developers.google.com/identity/images/g-logo.png",
        redirect_uri=config["redirect_uri"],
        scope=scope,
        key="google_oauth",
        use_container_width=True
    )
    
    return result

def get_user_info(access_token: str) -> dict:
    """Fetch user information from Google using access token."""
    try:
        response = requests.get(
            'https://www.googleapis.com/oauth2/v2/userinfo',
            headers={'Authorization': f'Bearer {access_token}'},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get user info: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching user info: {e}")
        return None

def clear_user_session():
    """Clear all user session data."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "chat_history": [],
        "user_email": "",
        "user_name": "",
        "user_grade": "",
        "authenticated": False,
        "setup_complete": False,
        "thread_id": str(uuid.uuid4()),
        "agent_memory": [],
        "feedback_data": [],  # Will be properly initialized by initialize_feedback_data()
        "show_feedback_popup": False,
        "current_feedback_index": -1,
        "pending_feedback": False,
        "feedback_locked": [],
        "temp_feedback": {"type": None, "reason": ""},
        "show_feedback_summary": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
   

def save_feedback_session():
    """Update the most recent AI message with feedback."""
    if not st.session_state.feedback_data or len(st.session_state.feedback_data) < 2:
        print("Warning: feedback_data not properly initialized")
        return
    
    # Find the most recent AI message in feedback_data and update it
    memory = st.session_state.feedback_data[1]['memory']
    
    # Find the last AI message that doesn't have feedback yet
    for i in range(len(memory) - 1, -1, -1):
        msg = memory[i]
        if (msg.get("type") == "AIMessage" and 
            msg.get("feedback_type") is None):
            
            # Update this message with feedback
            msg["feedback_type"] = st.session_state.temp_feedback["type"]
            msg["feedback_reason"] = st.session_state.temp_feedback["reason"]
            
            print(f"Updated feedback for AI message at index {i}")
            print(f"Feedback type: {msg['feedback_type']}")
            print(f"Feedback reason: {msg['feedback_reason']}")
            break
    else:
        print("Warning: No AI message found to update with feedback")
    
    # Reset temporary feedback
    st.session_state.temp_feedback = {"type": None, "reason": ""}
    st.session_state.pending_feedback = False
    
    print(f"Feedback session updated. Total messages in feedback_data: {len(memory)}")

def set_temp_feedback(feedback_type: str, reason: str = ""):
    """Set temporary feedback before saving."""
    st.session_state.temp_feedback = {
        "type": feedback_type,
        "reason": reason
    }

def render_feedback_popup(message_index: int):
    """Render the feedback popup for thumbs down positioned below the buttons."""
    # Create a container for the popup that appears right after the buttons
    with st.container():
        # Add some visual styling for the popup
        st.markdown("""
        <div style="
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #ff6b6b;
            margin: 10px 0;
        ">
        """, unsafe_allow_html=True)
        
        st.markdown("### 👎 Help us improve")
        st.markdown("Please tell us what went wrong:")
        
        # Feedback form
        with st.form(f"feedback_form_{message_index}", clear_on_submit=False):
            reason = st.text_area(
                "Reason for thumbs down:",
                placeholder="e.g., Incorrect information, Not helpful, Confusing explanation...",
                height=100,
                value=st.session_state.temp_feedback.get("reason", ""),
                key=f"feedback_reason_{message_index}"
            )
            
            col_submit, col_cancel = st.columns(2)
            
            with col_submit:
                submitted = st.form_submit_button("Submit Feedback", type="primary")
            
            with col_cancel:
                cancelled = st.form_submit_button("Cancel")
            
            if submitted:
                set_temp_feedback("thumbs_down", reason)
                st.session_state.show_feedback_popup = False
                st.session_state.current_feedback_index = -1
                st.success("Feedback recorded! Click 'Save' to finalize.")
                st.rerun()
            
            if cancelled:
                st.session_state.show_feedback_popup = False
                st.session_state.current_feedback_index = -1
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_feedback_buttons(message_index: int):
    """Render thumbs up, thumbs down, and save buttons for a message."""
    # Only show feedback buttons for the most recent AI message if feedback is pending
    is_most_recent_ai = False
    if st.session_state.chat_history:
        # Find the most recent AI message
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            sender, _ = st.session_state.chat_history[i]
            if sender == "AcadGenie":
                is_most_recent_ai = (i == message_index)
                break
    
    # Check if this message feedback is locked
    is_locked = message_index in st.session_state.feedback_locked
    
    if is_locked:
        # Show locked feedback state
        col1, col2 = st.columns([10, 2])
        with col2:
            # Get feedback type from locked feedback (you might want to store this differently)
            feedback_type = st.session_state.temp_feedback.get("type")  # This might not work as expected
            
            if feedback_type == "thumbs_up":
                st.markdown("✅ 👍")
            elif feedback_type == "thumbs_down":
                st.markdown("✅ 👎")
    elif is_most_recent_ai and st.session_state.pending_feedback:
        # Show interactive feedback buttons only for the most recent AI message when feedback is pending
        current_selection = st.session_state.temp_feedback["type"]
        
        col1, col2, col3, col4 = st.columns([8, 1, 1, 2])
        
        # Determine if feedback has been selected
        feedback_selected = current_selection is not None
        
        with col2:
            # Thumbs up button
            thumbs_up_pressed = st.button(
                "👍", 
                key=f"thumbs_up_{message_index}", 
                help="This response was helpful"
            )
            if thumbs_up_pressed:
                set_temp_feedback("thumbs_up")
                # Close any open popup
                st.session_state.show_feedback_popup = False
                st.session_state.current_feedback_index = -1
                st.rerun()
        
        with col3:
            # Thumbs down button
            thumbs_down_pressed = st.button(
                "👎", 
                key=f"thumbs_down_{message_index}", 
                help="This response needs improvement"
            )
            if thumbs_down_pressed:
                # Toggle popup for this specific message
                if (st.session_state.show_feedback_popup and 
                    st.session_state.current_feedback_index == message_index):
                    # Close popup if it's already open for this message
                    st.session_state.show_feedback_popup = False
                    st.session_state.current_feedback_index = -1
                else:
                    # Open popup for this message
                    st.session_state.show_feedback_popup = True
                    st.session_state.current_feedback_index = message_index
                    set_temp_feedback("thumbs_down", "")  # Reset reason for new selection
                st.rerun()
        
        with col4:
            save_pressed = st.button(
                "💾 Save", 
                key=f"save_feedback_{message_index}",
                disabled=not feedback_selected,
                type="primary" if feedback_selected else "secondary"
            )
            if save_pressed and feedback_selected:
                save_feedback_session()
                st.session_state.feedback_locked.append(message_index)
                # Close popup after saving
                st.session_state.show_feedback_popup = False
                st.session_state.current_feedback_index = -1
                st.success("Feedback saved successfully!")
                st.rerun()
        
        # Show current selection with visual indicators
        if feedback_selected:
            with col1:
                if current_selection == "thumbs_up":
                    st.markdown("**🔵 Selected:** 👍 Helpful")
                elif current_selection == "thumbs_down":
                    reason = st.session_state.temp_feedback.get("reason", "")
                    if reason:
                        st.markdown(f"**🔵 Selected:** 👎 Needs improvement - {reason[:50]}...")
                    else:
                        st.markdown("**🔵 Selected:** 👎 Needs improvement (click 👎 again to add reason)")
        
        # Render the feedback popup RIGHT HERE if it should be shown for this message
        if (st.session_state.show_feedback_popup and 
            st.session_state.current_feedback_index == message_index):
            render_feedback_popup(message_index)

def render_authentication():
    """Render the authentication page."""
    st.title("🤖 AcadGenie")
    st.markdown("Your AI learning companion. Please sign in to continue.")
    
    auth_result = authenticate_user()
    
    if auth_result and 'token' in auth_result:
        token_data = auth_result['token']
        
        # Extract access token
        access_token = None
        if isinstance(token_data, dict) and 'access_token' in token_data:
            access_token = token_data['access_token']
        elif isinstance(token_data, str):
            access_token = token_data
        
        if access_token:
            user_info = get_user_info(access_token)
            
            if user_info:
                st.session_state.user_email = user_info.get('email', '')
                st.session_state.user_name = user_info.get('name', '')
                st.session_state.authenticated = True
                st.success(f"Successfully logged in as {st.session_state.user_name}")
                st.rerun()
            else:
                st.error("Failed to get user information")
        else:
            st.error("Authentication failed - no access token received")

def render_grade_setup():
    """Render the grade selection page."""
    st.title("🤖 AcadGenie")
    st.markdown(f"Welcome, {st.session_state.user_name}!")
    st.markdown(f"Email: {st.session_state.user_email}")
    
    st.markdown("### Please select your grade level")
    
    with st.form("grade_setup"):
        grade = st.selectbox(
            "Your Grade:",
            options=["", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            index=0
        )
        
        submitted = st.form_submit_button("Start Learning")
        
        if submitted:
            if grade:
                st.session_state.user_grade = grade
                st.session_state.setup_complete = True
                st.rerun()
            else:
                st.error("Please select your grade to continue.")

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 AcadGenie")
    
    
    # Header with user info and controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"**{st.session_state.user_name}** | Grade {st.session_state.user_grade}")
    # with col2:
    #     if st.button("View Feedback", type="secondary"):
    #         st.session_state.show_feedback_summary = not st.session_state.get('show_feedback_summary', False)
    with col2:
        if st.button("Finish Chat", type="secondary"):
            #push feedback to db
            print("==============================Feedback===========================\n\n", st.session_state.feedback_data)
            push_feedback_data()
            
            # Clear session state
            st.session_state.chat_history = []
            st.session_state.agent_memory = []
            st.session_state.feedback_locked = []
            st.session_state.feedback_data = []
            st.session_state.pending_feedback = False
            st.session_state.temp_feedback = {"type": None, "reason": ""}
            st.rerun()
            
    with col3:
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.agent_memory = []
            st.session_state.feedback_data = []
            st.session_state.feedback_locked = []
            st.session_state.pending_feedback = False
            st.session_state.temp_feedback = {"type": None, "reason": ""}
            st.rerun()
    with col4:
        if st.button("Logout", type="secondary"):
            clear_user_session()
            st.rerun()
    
    # Show feedback summary if requested
    # if st.session_state.get('show_feedback_summary', False):
    #     with st.expander("📊 Feedback Summary", expanded=True):
    #         if st.session_state.feedback_data and len(st.session_state.feedback_data) >= 2:
    #             # Get session info from index 0
    #             session_info = st.session_state.feedback_data[0]
    #             # Get memory from index 1
    #             memory_object = st.session_state.feedback_data[1]
    #             memory = memory_object.get("memory", [])
                
    #             st.write(f"**Session ID:** {session_info.get('session_id', 'N/A')}")
    #             st.write(f"**User:** {session_info.get('user_name', 'N/A')} (Grade {session_info.get('user_grade', 'N/A')})")
    #             st.write(f"**Timestamp:** {session_info.get('timestamp', 'N/A')}")
                
    #             # Count feedback types from memory
    #             thumbs_up_count = 0
    #             thumbs_down_count = 0
    #             recent_improvements = []
                
    #             for msg in memory:
    #                 if msg.get("type") == "AIMessage" and "feedback_type" in msg:
    #                     if msg["feedback_type"] == "thumbs_up":
    #                         thumbs_up_count += 1
    #                     elif msg["feedback_type"] == "thumbs_down":
    #                         thumbs_down_count += 1
    #                         if msg.get("feedback_reason"):
    #                             recent_improvements.append(msg["feedback_reason"])
                
    #             col_up, col_down = st.columns(2)
    #             with col_up:
    #                 st.metric("👍 Positive", thumbs_up_count)
    #             with col_down:
    #                 st.metric("👎 Needs Improvement", thumbs_down_count)
                
    #             if recent_improvements:
    #                 st.markdown("**Recent improvement suggestions:**")
    #                 for reason in recent_improvements[-3:]:  # Show last 3
    #                     st.markdown(f"- {reason}")
                        
    #             st.write(f"**Total messages in memory:** {len(memory)}")
    #         else:
    #             st.info("No feedback data available yet.")
    
    st.markdown("---")
    
    # Prepare config for agent
    config = {
        "configurable": {
            "user_name": st.session_state.user_name,
            "grade": st.session_state.user_grade,
            "thread_id": st.session_state.thread_id
        }
    }
    
    # Chat input - only allow if no pending feedback
    if not st.session_state.pending_feedback:
        user_input = st.chat_input("Type your message here...")
    else:
        st.chat_input("Please provide feedback on the previous response before continuing...", disabled=True)
        user_input = None
    
    if user_input:
        # Add user message to display history
        st.session_state.chat_history.append(("You", user_input))
        
        with st.spinner("AcadGenie is thinking..."):
            try:
                # Get response from agent
                result = get_agent_response(user_input, config)
                
                # Parse and format the response
                ai_response = result.get("acadgenie", "I'm not sure how to respond!")
                parsed_response = parse_response(ai_response)
                print("Agent Response: ", parsed_response)
                # Format message for display
                formatted_msg = format_ai_message(parsed_response)
                
                # Add to display history
                st.session_state.chat_history.append(("AcadGenie", formatted_msg))
                
                # Set pending feedback for the new AI message
                st.session_state.pending_feedback = True
                
            except Exception as e:
                error_msg = "I encountered an error while processing your request. Please try again."
                st.session_state.chat_history.append(("AcadGenie", error_msg))
                st.session_state.pending_feedback = True  # Also require feedback for error messages
                st.error(f"Error: {str(e)}")
    
    # Display chat history with feedback buttons
    for i, (sender, message) in enumerate(st.session_state.chat_history):
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.markdown(message)
            
            # Add feedback buttons only for AcadGenie messages
            if sender == "AcadGenie":
                render_feedback_buttons(i)
    
    # Show pending feedback warning if needed
    if st.session_state.pending_feedback and st.session_state.temp_feedback["type"] is None:
        st.warning("⚠️ Please provide feedback on the above response before sending another message.")

def format_ai_message(parsed_response: dict) -> str:
    """Format AI response for display."""
    if isinstance(parsed_response, dict):
        formatted_msg = parsed_response.get("conversation_message", "")
        
        # Handle structured question data if present
        question_data = parsed_response.get("question_data")
        if question_data:
            formatted_msg += f"\n\n**Question:** {question_data.get('question', '')}\n"
            formatted_msg += "\n**Options:**\n"
            for option in question_data.get('options', []):
                option_letter = option.get('option', '')
                option_text = option.get('text', '')
                formatted_msg += f"- {option_letter}: {option_text}\n"
            
            formatted_msg += f"\n**Correct Answer:** {question_data.get('correct_answer', '')}"
            formatted_msg += f"\n**Explanation:** {question_data.get('explanation', '')}"
            
            comment = question_data.get('comment')
            if comment:
                formatted_msg += f"\n**Comment:** {comment}"
        
        return formatted_msg if formatted_msg else str(parsed_response)
    else:
        return str(parsed_response)

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="AcadGenie Chat",
        page_icon="🧠",
        layout="centered"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Route to appropriate page based on authentication state
    if not st.session_state.authenticated:
        render_authentication()
    elif not st.session_state.setup_complete:
        render_grade_setup()
    else:
        initialize_feedback_data()
        render_chat_interface()

if __name__ == "__main__":
    main()
