from dotenv import load_dotenv
import streamlit as st
import json
import traceback
import uuid
import asyncio
from typing import TypedDict, Annotated, Any
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from datetime import datetime
from pymongo import MongoClient
import os

from questions import getQuestions
load_dotenv()

sys_prompt = """
You are AcadGenie, an expert academic assistant specializing in teaching through highly interactive, guided practice. Your role is to guide learners through complex concepts by breaking them down into small, manageable steps and using a variety of interactive prompts such as multiple-choice questions (MCQs), fill-in-the-blanks (FIBs), and short answers. You do not solve problems outright but help learners solve them through their own thinking, rooted in Socratic questioning, error analysis, and constructivist learning. You adjust the complexity of your responses and generated questions based on the learner’s grade level and focus on fostering true conceptual clarity and real-world application.
You act as a Socratic remediation tutor to help them understand where they are struggling and how to improve their understanding of the concept. 

## Instructions
Do not give direct answers.
Instead, begin by probing the learner’s understanding through carefully designed questions.
Convert every user query—no matter how simple—into a sequence of conceptual, diagnostic MCQs or guided questions.

Your objectives are: 
- Identify and build on the learner’s prior knowledge. 
- Elicit misconceptions using plausible distractors. 
- Guide reasoning step-by-step with Socratic logic.
- Avoid procedural explanation unless the learner constructs it themselves.

Constraints:
No thinking aloud, no meta-commentary.
Do not solve the problem.
Only respond with tightly constructed, single-question prompts (MCQ or short answer).
End each prompt with a single question only. Do not list multiple questions at once.


## Interaction Flow

Evaluate User Input:
	- If the solution to the user's question is fact, definitions, simple statements, etc. , respond with simple explanation enough to make student understand the question with the help of solution.
	- If the solution to the user's question is a question having any procedure, process, theorem, algorithm, etc. , decompose it into sequential sub-problems or steps (e.g. identifying inputs and methods to use, recalling formulas, applying formulas, etc.).

Decomposing and Handling sequential sub-problems or steps:
	- For each step, generate an interactive Multiple Choice Question(MCQ) to guide the learner:
	- Always include plausible options with distractor rationales (DRs) to address common misconceptions.
    - These question stem of MCQs can be of multiple forms like "Fill in the blank", "Short Answer", etc.
    - Vary the question type across steps to avoid repetition (e.g., don't use same type of MCQs multiple times for similar steps).
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
-When there is not question data, output question_data as empty object e.g. "question_data": { }
-Make sure all the outputs follow below given output structure:

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

Important Notes
    - Always break down a question based on the learner's grade level.
	- Adjust the complexity of generated questions based on the learner's grade level.
	- Do not repeat the same type of question multiple times for similar steps.
	- Do not solve any step for the learner; always guide them to think through it.
	- Provide feedback that is specific and tied to conceptual clarity.
	- Never rush to solve the full problem; focus on one step at a time.
"""


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
        uri = os.environ["db_uri"]
        client = MongoClient(uri)
        db = client[os.environ["db_name"]]
        collection = db[os.environ["collection_name"]]
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
    
    system_msg = f"{sys_prompt} Address the user as {user_name} from grade {grade}."
    
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
                model="gpt-4.1",
                temperature=0.3,
                api_key=os.environ["OPENAI_API_KEY"]
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

def clear_user_session():
    """Clear all user session data."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "chat_history": [],
        "user_name": "",
        "user_grade": "",
        "authenticated": False,
        "setup_complete": False,
        "thread_id": str(uuid.uuid4()),
        "agent_memory": [],
        "feedback_data": [],  # Will be properly initialized by initialize_feedback_data()
        "show_feedback_popup": False,
        "current_feedback_index": -1,
        "feedback_locked": [],
        "temp_feedback": {"type": None, "reason": ""},
        "show_feedback_summary": False,
        "pending_feedback_save": {},  # Track which messages have unsaved feedback
        # New states for question selection
        "selected_board": "",
        "selected_grade": "",
        "selected_subject": "",
        "available_questions": [],
        "questions_loaded": False,
        "loading_questions": False,
        "selected_question": "",
        "pending_question": "" 
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

async def fetch_questions_async(board: str, grade: str, subject: str):
    """Wrapper to call the async getQuestions function."""
    return await getQuestions(board, grade, subject)

def save_feedback_session(message_index: int):
    """Update the specific AI message with feedback."""
    if not st.session_state.feedback_data or len(st.session_state.feedback_data) < 2:
        print("Warning: feedback_data not properly initialized")
        return
    print("\n\n\nSave Feedback method called.....\n\n\n")
    # Get the feedback for this specific message
    feedback_info = st.session_state.pending_feedback_save.get(message_index, {})
    feedback_type = feedback_info.get("type")
    feedback_reason = feedback_info.get("reason", "")
    
    if not feedback_type:
        return
    
    # We need to map the chat history index to the feedback_data index
    memory = st.session_state.feedback_data[1]['memory']
    
    # Find the corresponding AI message in memory (counting backwards from the end)
    for i in range(len(memory) - 1, -1, -1):
        msg = memory[i]
        if msg.get("type") == "AIMessage":
            # Update this message with feedback
            msg["feedback_type"] = feedback_type
            msg["feedback_reason"] = feedback_reason
            
            print(f"Updated feedback for AI message at memory index {i}")
            print(f"Feedback type: {msg['feedback_type']}")
            print(f"Feedback reason: {msg['feedback_reason']}")
            break
    else:
        print("Warning: No corresponding AI message found in feedback_data")
    
    # Remove from pending feedback save
    if message_index in st.session_state.pending_feedback_save:
        del st.session_state.pending_feedback_save[message_index]
    
    # Add to locked feedback
    if message_index not in st.session_state.feedback_locked:
        st.session_state.feedback_locked.append(message_index)
    
    print(f"Feedback session updated for message {message_index}. Total messages in feedback_data: {len(memory)}")

def set_temp_feedback(message_index: int, feedback_type: str, reason: str = ""):
    """Set temporary feedback for a specific message."""
    st.session_state.pending_feedback_save[message_index] = {
        "type": feedback_type,
        "reason": reason
    }

def has_unsaved_feedback():
    """Check if there are any unsaved feedback selections."""
    return len(st.session_state.pending_feedback_save) > 0

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
        
        # Get current reason if any
        current_feedback = st.session_state.pending_feedback_save.get(message_index, {})
        current_reason = current_feedback.get("reason", "")
        
        # Feedback form
        with st.form(f"feedback_form_{message_index}", clear_on_submit=False):
            reason = st.text_area(
                "Reason for thumbs down:",
                placeholder="e.g., Incorrect information, Not helpful, Confusing explanation...",
                height=100,
                value=current_reason,
                key=f"feedback_reason_{message_index}"
            )
            
            col_submit, col_cancel = st.columns(2)
            
            with col_submit:
                submitted = st.form_submit_button("Submit Feedback", type="primary")
            
            with col_cancel:
                cancelled = st.form_submit_button("Cancel")
            
            if submitted:
                if reason.strip():  # Ensure reason is provided for thumbs down
                    set_temp_feedback(message_index, "thumbs_down", reason.strip())
                    st.session_state.show_feedback_popup = False
                    st.session_state.current_feedback_index = -1
                    st.success("Feedback recorded! Click 'Save' to finalize.")
                    st.rerun()
                else:
                    st.error("Please provide a reason for the thumbs down feedback.")
            
            if cancelled:
                # Remove any pending feedback for this message
                if message_index in st.session_state.pending_feedback_save:
                    del st.session_state.pending_feedback_save[message_index]
                st.session_state.show_feedback_popup = False
                st.session_state.current_feedback_index = -1
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_feedback_buttons(message_index: int):
    """Render thumbs up, thumbs down, and save buttons for a message."""
    
    # Find the index of the most recent AI message
    most_recent_ai_index = -1
    for i in range(len(st.session_state.chat_history) - 1, -1, -1):
        if i < len(st.session_state.chat_history):
            sender, _ = st.session_state.chat_history[i]
            if sender == "AcadGenie":
                most_recent_ai_index = i
                break
    
    # Check if this message feedback is locked
    is_locked = message_index in st.session_state.feedback_locked
    
    if is_locked:
        # Show locked feedback state
        col1, col2 = st.columns([10, 2])
        with col2:
            # Get the saved feedback type from feedback_data
            # Find corresponding feedback in the data
            memory = st.session_state.feedback_data[1]['memory']
            ai_message_count = 0
            for i in range(message_index + 1):
                if i < len(st.session_state.chat_history):
                    sender, _ = st.session_state.chat_history[i]
                    if sender == "AcadGenie":
                        ai_message_count += 1
            
            # Find the feedback type
            feedback_type = None
            ai_messages_found = 0
            for i in range(len(memory)):
                msg = memory[i]
                if msg.get("type") == "AIMessage":
                    ai_messages_found += 1
                    if ai_messages_found == ai_message_count:
                        feedback_type = msg.get("feedback_type")
                        break
            
            if feedback_type == "thumbs_up":
                st.markdown("✅ 👍")
            elif feedback_type == "thumbs_down":
                st.markdown("✅ 👎")
    elif message_index == most_recent_ai_index:
        # Only show interactive feedback buttons for the most recent AI message
        current_selection = st.session_state.pending_feedback_save.get(message_index, {})
        feedback_type = current_selection.get("type")
        
        col1, col2, col3, col4 = st.columns([8, 1, 1, 2])
        
        # Determine if feedback has been selected for this message
        feedback_selected = feedback_type is not None
        
        with col2:
            # Thumbs up button
            thumbs_up_pressed = st.button(
                "👍", 
                key=f"thumbs_up_{message_index}", 
                help="This response was helpful"
            )
            if thumbs_up_pressed:
                set_temp_feedback(message_index, "thumbs_up")
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
                    # Initialize thumbs down feedback if not already set
                    if feedback_type != "thumbs_down":
                        set_temp_feedback(message_index, "thumbs_down", "")
                st.rerun()
        
        with col4:
            save_pressed = st.button(
                "💾 Save", 
                key=f"save_feedback_{message_index}",
                disabled=not feedback_selected,
                type="primary" if feedback_selected else "secondary"
            )
            if save_pressed and feedback_selected:
                # For thumbs down, ensure reason is provided
                if (feedback_type == "thumbs_down" and 
                    not current_selection.get("reason", "").strip()):
                    st.error("Please provide a reason for thumbs down feedback.")
                else:
                    save_feedback_session(message_index)
                    # Close popup after saving
                    st.session_state.show_feedback_popup = False
                    st.session_state.current_feedback_index = -1
                    st.success("Feedback saved successfully!")
                    st.rerun()
        
        # Show current selection with visual indicators
        if feedback_selected:
            with col1:
                if feedback_type == "thumbs_up":
                    st.markdown("**🔵 Selected:** 👍 Helpful")
                elif feedback_type == "thumbs_down":
                    reason = current_selection.get("reason", "")
                    if reason:
                        st.markdown(f"**🔵 Selected:** 👎 Needs improvement - {reason}...")
                    else:
                        st.markdown("**🔵 Selected:** 👎 Needs improvement (please add reason)")
        
        # Render the feedback popup RIGHT HERE if it should be shown for this message
        if (st.session_state.show_feedback_popup and 
            st.session_state.current_feedback_index == message_index):
            render_feedback_popup(message_index)

def render_question_selection_sidebar():
    """Render the question selection sidebar."""
    with st.sidebar:
        st.header("📚 Question Selection")
        
        # Board selection
        board = st.selectbox(
            "Board:",
            options=["", "CBSE", "ICSE"],
            index=0,
            key="board_select"
        )
        
        # Grade selection
        grade = st.selectbox(
            "Grade:",
            options=["", "6", "7", "8", "9", "10"],
            index=0,
            key="grade_select"
        )
        
        # Subject selection
        subject = st.selectbox(
            "Subject:",
            options=["", "Mathematics", "Physics", "Chemistry", "Biology"],
            index=0,
            key="subject_select"
        )
        
        # Submit button
        if st.button("🔍 Get Questions", type="primary", key="get_questions_btn"):
            if board and grade and subject:
                st.session_state.selected_board = board
                st.session_state.selected_grade = grade
                st.session_state.selected_subject = subject
                st.session_state.loading_questions = True
                st.session_state.questions_loaded = False
                st.rerun()
            else:
                st.error("Please select board, grade, and subject.")
        
        # Show loading state
        if st.session_state.loading_questions:
            with st.spinner("Loading questions..."):
                try:
                    # Run the async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    questions = loop.run_until_complete(
                        fetch_questions_async(
                            st.session_state.selected_board,
                            st.session_state.selected_grade,
                            st.session_state.selected_subject
                        )
                    )
                    loop.close()
                    
                    st.session_state.available_questions = questions or []
                    st.session_state.questions_loaded = True
                    st.session_state.loading_questions = False
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading questions: {str(e)}")
                    st.session_state.loading_questions = False
                    st.rerun()
        
        # Show questions dropdown if loaded
        if st.session_state.questions_loaded and st.session_state.available_questions:
            st.markdown("---")
            st.subheader("📝 Available Questions")
            
            # Inject custom CSS for styling
            st.markdown("""
            <style>
            .question-correct {
                border-left: 4px solid #28a745 !important;
                background-color: #f8fff9 !important;
                padding: 8px !important;
                margin: 2px 0 !important;
                border-radius: 4px !important;
            }
            .question-incorrect {
                border-left: 4px solid #dc3545 !important;
                background-color: #fff8f8 !important;
                padding: 8px !important;
                margin: 2px 0 !important;
                border-radius: 4px !important;
            }
            .question-default {
                border-left: 4px solid #6c757d !important;
                background-color: #f8f9fa !important;
                padding: 8px !important;
                margin: 2px 0 !important;
                border-radius: 4px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create options for the selectbox
            # question_options = ["Select a question..."] + [
            #     f"Q{i+1}: {q[:50]}..." if len(q) > 50 else f"Q{i+1}: {q}"
            #     for i, q in enumerate(st.session_state.available_questions)
            # ]
            
            # Create options for the selectbox with color indicators
            question_options = ["Select a question..."] + [
                f"{'✅' if q.get('is_correct', False) else '❌'} Q{i+1}: {q.get('question', str(q))[:50]}..." 
                if len(str(q.get('question', q))) > 50 
                else f"{'✅' if q.get('is_correct', False) else '❌'} Q{i+1}: {q.get('question', str(q))}"
                for i, q in enumerate(st.session_state.available_questions)
            ]
            
            selected_question_index = st.selectbox(
                "Choose a question:",
                options=range(len(question_options)),
                format_func=lambda x: question_options[x],
                key="question_select"
            )
            
            # If a question is selected, handle it
            if selected_question_index > 0:  # 0 is "Select a question..."
                actual_index = selected_question_index - 1
                selected_question = st.session_state.available_questions[actual_index]
                
                if st.button("📤 Send Question to Chat", key="send_question_btn"):
                    # Add the selected question to chat history
                    # st.session_state.chat_history.append(("You", selected_question))
                    st.session_state.pending_question = selected_question
                    # Clear the question selection
                    st.session_state.selected_question = ""
                    st.rerun()
                
                # Show preview of selected question
                with st.expander("📖 Question Preview"):
                    st.write(selected_question)
        
        elif st.session_state.questions_loaded and not st.session_state.available_questions:
            st.warning("No questions found for the selected criteria.")
        
        # Show current selection summary if any
        if st.session_state.selected_board:
            st.markdown("---")
            st.markdown("**Current Selection:**")
            st.markdown(f"🏫 **Board:** {st.session_state.selected_board}")
            st.markdown(f"📚 **Grade:** {st.session_state.selected_grade}")
            st.markdown(f"🔬 **Subject:** {st.session_state.selected_subject}")
            
            if st.button("🔄 Reset Selection", key="reset_selection_btn"):
                st.session_state.selected_board = ""
                st.session_state.selected_grade = ""
                st.session_state.selected_subject = ""
                st.session_state.available_questions = []
                st.session_state.questions_loaded = False
                st.session_state.loading_questions = False
                st.rerun()

def render_name_input():
    """Render the name input page."""
    st.title("🤖 AcadGenie")
    st.markdown("Your AI learning companion. Please enter your details to continue.")
    
    with st.form("name_input_form"):
        st.markdown("### Enter your information")
        
        name = st.text_input(
            "Your Name:",
            placeholder="Enter your full name",
            value=""
        )
        
        grade = st.selectbox(
            "Your Grade:",
            options=["", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            index=0
        )
        
        submitted = st.form_submit_button("Start Learning", type="primary")
        
        if submitted:
            if name.strip() and grade:
                st.session_state.user_name = name.strip()
                st.session_state.user_grade = grade
                st.session_state.authenticated = True
                st.session_state.setup_complete = True
                st.success(f"Welcome, {st.session_state.user_name}!")
                st.rerun()
            else:
                if not name.strip():
                    st.error("Please enter your name.")
                if not grade:
                    st.error("Please select your grade.")

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 AcadGenie")
    
    
    # Header with user info and controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"**{st.session_state.user_name}** | Grade {st.session_state.user_grade}")
    
    with col2:
        if st.button("Save Chat", type="primary"):
            #push feedback to db
            print("==============================Feedback===========================\n\n", st.session_state.feedback_data)
            push_feedback_data()
            
            # Clear session state
            st.session_state.chat_history = []
            st.session_state.agent_memory = []
            st.session_state.feedback_locked = []
            st.session_state.feedback_data = []
            st.session_state.pending_feedback_save = {}
            st.rerun()
            
    with col3:
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.agent_memory = []
            st.session_state.feedback_data = []
            st.session_state.feedback_locked = []
            st.session_state.pending_feedback_save = {}
            st.rerun()
        
    with col4:
        if st.button("Start Over", type="secondary"):
            clear_user_session()
            st.rerun()
    
    review_role = st.radio(
        label="Choose your review perspective:",
        options=["teacher", "student"],
        index=0,  # Default to "Teacher"
        key="review_role_select",
        horizontal=True,  # Display options horizontally
        # label_visibility="collapsed"  # Hide the label since we have custom text above
    )   
    
    # Store the selected role in session state
    st.session_state.review_role = review_role
    
    st.markdown("---")
    
    # Prepare config for agent
    config = {
        "configurable": {
            "user_name": st.session_state.user_name,
            "grade": st.session_state.user_grade,
            "thread_id": st.session_state.thread_id
        }
    }
    
    # Check for unsaved feedback and show warning
    if has_unsaved_feedback():
        st.warning("⚠️ You have unsaved feedback. Please save your feedback before it gets lost, or continue chatting without saving.")
    
    # Chat input - always enabled now (feedback is optional)
    user_input = st.chat_input("Type your message here...")
    
    # Check for pending question from sidebar
    pending_question = st.session_state.get("pending_question", "")
    if pending_question:
        # Process the pending question like regular user input
        st_chat_input = pending_question
        st.session_state.pending_question = ""

        if st_chat_input:
            user_input = f"""Here is a question which I want to understand better using step-by-step reasoning.
            Along with question, options originally provided with the question, option chosen by student, solution to the provided question is also provided.
            
            Question: {st_chat_input['question']}
            Options: {st_chat_input['options']}
            Solution: {st_chat_input['solution']}
            Chosen_Option: {st_chat_input['chosen_option']}.
            """
            # Make sure the output is properly formatted. Use inline latex for mathematical expressions and use `\\(` an d `\\)` to enclose it."""
            
    if user_input:
        # Add user message to display history
        st.session_state.chat_history.append(("You", user_input))
        
    chat_container = st.container(height=400) 
    # Display chat history with feedback buttons
    with chat_container:
        for i, (sender, message) in enumerate(st.session_state.chat_history):
            with st.chat_message("user" if sender == "You" else "assistant"):
                st.markdown(message)
                
                # Add feedback buttons only for AcadGenie messages
                if sender == "AcadGenie":
                    render_feedback_buttons(i)
        
    if user_input:
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
                st.rerun()
                
            except Exception as e:
                error_msg = "I encountered an error while processing your request. Please try again."
                st.session_state.chat_history.append(("AcadGenie", error_msg))
                st.error(f"Error: {str(e)}")
                st.rerun()

def format_ai_message(parsed_response: dict) -> str:
    """Format AI response for display."""
    
    selected_role = st.session_state.get('review_role', 'Teacher')
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
        
            if selected_role == "teacher":  
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
    if not st.session_state.authenticated or not st.session_state.setup_complete:
        render_name_input()
    else:
        initialize_feedback_data()
        render_question_selection_sidebar()
        render_chat_interface()

if __name__ == "__main__":
    main()