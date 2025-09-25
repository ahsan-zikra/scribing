# graph.py  (single file, summary included)
from __future__ import annotations

import os
import sys
import time
import asyncio
from dataclasses import dataclass
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from pydantic import BaseModel

# --------------------------------------------------
# 1.  Env loading
# --------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for env in ("../.env.local", "./.env.local", "./.env"):
    load_dotenv(env)

# --------------------------------------------------
# 2.  Pydantic I/O schemas
# --------------------------------------------------
class Input(BaseModel):
    raw_scribe: str

class Output(BaseModel):
    insights: List[str]
    probing_questions: List[str]
    chat_note: str
    red_flags: List[str]
    summary: str

class InsightsOutput(BaseModel):
    insights: List[str]
    red_flags: List[str]

class ProbingQuestionsOutput(BaseModel):
    probing_questions: List[str]

class ChatNoteOutput(BaseModel):
    chat_note: str

class SummaryOutput(BaseModel):
    summary: str

# --------------------------------------------------
# 3.  Prompts
# --------------------------------------------------
INSIGHTS_PROMPT = """
You are a medical assistant tasked with extracting key insights and identifying red flags from a doctor-patient conversation transcript. 

**Task:**
- Analyse the conversation to identify key medical facts, symptoms, conditions, or concerns mentioned by the patient.
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
"""

PROBING_QUESTIONS_PROMPT = """
You are a medical assistant tasked with generating probing questions based on a doctor-patient conversation transcript and extracted insights. 

**Task:**
- Based on the provided conversation and insights, generate follow-up questions for the doctor to ask the patient.
- Ensure questions are specific, clarify ambiguous points, and are appropriate for a clinical context.
- Do not repeat questions already asked by the doctor in the conversation.
- Use the insights to guide question generation.
- **PRIORITISE RED FLAGS**: If red flags are present, generate questions that help assess the urgency and severity of those concerns.

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
- Summarise the key medical information, symptoms, and concerns discussed in the conversation.
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

SUMMARY_PROMPT = """
You are a medical assistant.  
Write a **single-paragraph plain-language summary** (2-3 sentences) of the doctor-patient conversation below.  
The summary must be suitable for the patient to read and should **highlight any red-flagged urgent issues** first.

Conversation:
{conversation}

Insights:
{insights}

Red flags (if any):
{red_flags}

Chat note:
{chat_note}
"""

# --------------------------------------------------
# 4.  LangGraph state
# --------------------------------------------------
@dataclass
class State(TypedDict):
    raw_scribe: str
    insights: List[str]
    probing_questions: List[str]
    chat_note: str
    red_flags: List[str]
    summary: str

# --------------------------------------------------
# 5.  Node functions
# --------------------------------------------------
def insights_node(state: State) -> State:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(InsightsOutput)
        prompt = PromptTemplate.from_template(INSIGHTS_PROMPT)
        prompt_value = prompt.invoke({"conversation": state["raw_scribe"]})
        llm_response = structured_llm.invoke(prompt_value)
        state["insights"] = llm_response.insights
        state["red_flags"] = llm_response.red_flags
    except Exception:
        state["insights"] = []
        state["red_flags"] = []
    return state

def probing_questions_node(state: State) -> State:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(ProbingQuestionsOutput)
        prompt = PromptTemplate.from_template(PROBING_QUESTIONS_PROMPT)
        prompt_value = prompt.invoke({
            "conversation": state["raw_scribe"],
            "insights": state["insights"],
            "red_flags": state["red_flags"],
        })
        llm_response = structured_llm.invoke(prompt_value)
        state["probing_questions"] = llm_response.probing_questions
    except Exception:
        state["probing_questions"] = []
    return state

def chat_note_node(state: State) -> State:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(ChatNoteOutput)
        prompt = PromptTemplate.from_template(CHAT_NOTE_PROMPT)
        prompt_value = prompt.invoke({
            "conversation": state["raw_scribe"],
            "insights": state["insights"],
            "red_flags": state["red_flags"],
        })
        llm_response = structured_llm.invoke(prompt_value)
        state["chat_note"] = llm_response.chat_note
    except Exception:
        state["chat_note"] = ""
    return state

def summary_node(state: State) -> State:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(SummaryOutput)
        prompt = PromptTemplate.from_template(SUMMARY_PROMPT)
        prompt_value = prompt.invoke({
            "conversation": state["raw_scribe"],
            "insights": state["insights"],
            "red_flags": state["red_flags"],
            "chat_note": state["chat_note"],
        })
        state["summary"] = structured_llm.invoke(prompt_value).summary
    except Exception:
        state["summary"] = "Summary could not be generated."
    return state

# --------------------------------------------------
# 6.  Build and compile graph
# --------------------------------------------------
builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node("insights", insights_node)
builder.add_node("probing_questions", probing_questions_node)
builder.add_node("chat_note", chat_note_node)
builder.add_node("summary", summary_node)

builder.add_edge(START, "insights")
builder.add_edge("insights", "probing_questions")
builder.add_edge("probing_questions", "chat_note")
builder.add_edge("chat_note", "summary")
builder.set_finish_point("summary")

Graph = builder.compile()

# --------------------------------------------------
# 7.  Async wrapper (keeps old interface)
# --------------------------------------------------
async def ainvoke(input_dict: dict) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, Graph.invoke, input_dict)

Graph.ainvoke = ainvoke

# --------------------------------------------------
# 8.  Quick sanity test
# --------------------------------------------------
if __name__ == "__main__":
    t0 = time.process_time()
    result = Graph.invoke({"raw_scribe": "Patient reports chest pain and shortness of breath."})
    print(result)
    t1 = time.process_time()
    print("Time taken:", t1 - t0, "seconds")
