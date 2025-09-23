from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, List
from langgraph.graph import StateGraph, START
from langgraph.runtime import Runtime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv('../.env.local')

class Input(BaseModel):
    """Input schema for the agent."""
    raw_scribe: str

class Output(BaseModel):
    """Output schema for the agent."""
    insights: List[str]
    probing_questions: List[str]
    chat_note: str
    red_flags: List[str]

class InsightsOutput(BaseModel):
    """Output schema for the insights node."""
    insights: List[str]
    red_flags: List[str]

class ProbingQuestionsOutput(BaseModel):
    """Output schema for the probing questions node."""
    probing_questions: List[str]

class ChatNoteOutput(BaseModel):
    """Output schema for the chat note node."""
    chat_note: str

# Define separate prompts for each node
INSIGHTS_PROMPT = """
You are a medical assistant tasked with extracting key insights and identifying red flags from a doctor-patient conversation transcript. 

**Task:**
- Analyze the conversation to identify key medical facts, symptoms, conditions, or concerns mentioned by the patient.
- Consider emotional, psychological, or social factors that might be relevant.
- Return a list of concise insights.
- **Identify RED FLAGS**: Look for urgent medical concerns that require immediate attention, such as:
  - Severe chest pain, difficulty breathing, or sudden neurological symptoms
  - Thoughts of self-harm or suicidal ideation
  - Acute bleeding or trauma
  - Signs of acute infection or sepsis
  - Any other potentially life-threatening conditions
- Return red flags as a separate list of concise statements.
Note: Never return empty lists even if no insights or red flags are found.

**Output structure:**
class InsightsOutput(BaseModel):
    insights: List[str]        # Regular medical insights and observations
    red_flags: List[str]       # Urgent concerns requiring immediate attention

**Conversation:**
{conversation}

**Instructions:**
- Insights should be factual observations from the conversation
- Red flags should only include truly urgent, potentially life-threatening issues
- If no red flags are present, return an empty list for red_flags
"""

PROBING_QUESTIONS_PROMPT = """
You are a medical assistant tasked with generating probing questions based on a doctor-patient conversation transcript and extracted insights. 

**Task:**
- Based on the provided conversation and insights, generate follow-up questions for the doctor to ask the patient.
- Ensure questions are specific, clarify ambiguous points, and are appropriate for a clinical context.
- Do not repeat questions already asked by the doctor in the conversation.
- Use the insights to guide question generation.
- **PRIORITIZE RED FLAGS**: If red flags are present, generate questions that help assess the urgency and severity of those concerns.

**Output structure:**
class ProbingQuestionsOutput(BaseModel):
    probing_questions: List[str]

**Conversation:**
{conversation}

**Insights:**
{insights}

**Red Flags (if any):**
{red_flags}
"""

CHAT_NOTE_PROMPT = """
You are a medical assistant tasked with generating a concise chat note (clinical summary) based on a doctor-patient conversation transcript and extracted insights.

**Task:**
- Summarize the key medical information, symptoms, and concerns discussed in the conversation.
- Incorporate the provided insights to ensure accuracy.
- **HIGHLIGHT RED FLAGS**: Clearly mark any red flags identified, as these require immediate attention.
- Write a concise chat note suitable for a medical record, focusing on clinical relevance.

**Output structure:**
class ChatNoteOutput(BaseModel):
    chat_note: str

**Conversation:**
{conversation}

**Insights:**
{insights}

**Red Flags (if any):**
{red_flags}

**Note:** Structure the chat note to clearly separate routine findings from any urgent red flags.
"""

# API key (replace with your actual key or load from .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@dataclass
class State(TypedDict):
    """State for the agent."""
    raw_scribe: str
    insights: List[str]
    probing_questions: List[str]
    chat_note: str
    red_flags: List[str]

async def insights_node(state: State) -> State:
    """Generates insights and red flags from the conversation."""
    try:
        print("Invoking Insights LLM")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(InsightsOutput)
        prompt = PromptTemplate.from_template(INSIGHTS_PROMPT)
        prompt_value = prompt.invoke({"conversation": state["raw_scribe"]})
        llm_response = structured_llm.invoke(prompt_value)
        state['insights'] = llm_response.insights
        state['red_flags'] = llm_response.red_flags
        print(f"Insights: {state['insights']}")
        print(f"Red Flags: {state['red_flags']}")
    except Exception as e:
        print(f"Error in insights node: {e}")
        state['insights'] = []
        state['red_flags'] = []
    return state

async def probing_questions_node(state: State) -> State:
    """Generates probing questions based on conversation, insights, and red flags."""
    try:
        print("Invoking Probing Questions LLM")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(ProbingQuestionsOutput)
        prompt = PromptTemplate.from_template(PROBING_QUESTIONS_PROMPT)
        prompt_value = prompt.invoke({
            "conversation": state["raw_scribe"],
            "insights": state["insights"],
            "red_flags": state["red_flags"]
        })
        llm_response = structured_llm.invoke(prompt_value)
        state['probing_questions'] = llm_response.probing_questions
        print(f"Probing Questions: {state['probing_questions']}")
    except Exception as e:
        print(f"Error in probing questions node: {e}")
        state['probing_questions'] = []
    return state

async def chat_note_node(state: State) -> State:
    """Generates a chat note based on conversation, insights, and red flags."""
    try:
        print("Invoking Chat Note LLM")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(ChatNoteOutput)
        prompt = PromptTemplate.from_template(CHAT_NOTE_PROMPT)
        prompt_value = prompt.invoke({
            "conversation": state["raw_scribe"],
            "insights": state["insights"],
            "red_flags": state["red_flags"]
        })
        llm_response = structured_llm.invoke(prompt_value)
        state['chat_note'] = llm_response.chat_note
        print(f"Chat Note: {state['chat_note']}")
    except Exception as e:
        print(f"Error in chat note node: {e}")
        state['chat_note'] = ""
    return state

# Build the graph
builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node("insights", insights_node)
builder.add_node("probing_questions", probing_questions_node)
builder.add_node("chat_note", chat_note_node)

# Define edges
builder.add_edge(START, "insights")
builder.add_edge("insights", "probing_questions")
builder.add_edge("probing_questions", "chat_note")
builder.set_finish_point("chat_note")

Graph = builder.compile()
