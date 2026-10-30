import os
from typing import TypedDict, Annotated, Literal
import operator

import psycopg
from psycopg_pool import ConnectionPool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_result: str
    hotel_result: str
    itinerary: str
    llm_calls: int

def intent_classifier(state: TravelState) -> Literal["travel_plan", "general_chat"]:
    """Classifies incoming user query to route between travel workflow and small talk/guardrails."""
    query = state["user_query"].strip()
    classification_prompt = f"""
    Classify the following user query into one of two categories:
    1. 'TRAVEL_PLAN': The user is specifically asking for travel planning, flight options, hotel suggestions, or itineraries.
    2. 'GENERAL_CHAT': Greetings (like 'hi', 'hello'), casual small talk, or non-travel prompts.

    User Query: "{query}"

    Respond with ONLY 'TRAVEL_PLAN' or 'GENERAL_CHAT'.
    """
    response = llm.invoke([
        SystemMessage(content="You are a classifier that strictly outputs either 'TRAVEL_PLAN' or 'GENERAL_CHAT'."),
        HumanMessage(content=classification_prompt)
    ]).content.strip().upper()

    if "TRAVEL_PLAN" in response:
        return "travel_plan"
    else:
        return "general_chat"

def general_chat_agent(state: TravelState):
    """Handles greetings and off-topic queries without triggering travel tools or hallucinating."""
    query = state["user_query"].strip()
    prompt = f"""
    You are an AI Travel Planning Assistant.
    If the user is greeting you (e.g., 'hi', 'hello'), respond warmly and invite them to share their travel destination, dates, or preferences.
    If the query is completely unrelated to travel, politely explain:
    "I am specialized in travel planning (flights, hotels, and itineraries). I cannot assist with unrelated topics, but feel free to ask me for travel recommendations!"

    User Input: "{query}"
    """
    response = llm.invoke([
        SystemMessage(content="You are a helpful travel assistant. Stick strictly to travel topics."),
        HumanMessage(content=prompt)
    ])
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_result": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def hotel_agent(state: TravelState):
    query = state["user_query"]
    hotel_data = tavily_search(query)
    return {
        "hotel_result": hotel_data,
        "messages": [
            AIMessage(content="Hotel fetched data")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def itinerary_agent(state: TravelState):
    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state["user_query"]}

    Flight Results:
    {state["flight_result"]}

    Hotel Results:
    {state["hotel_result"]}
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner. Create practical day-by-day itineraries."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def final_agent(state: TravelState):
    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state["flight_result"]}

    Hotels:
    {state["hotel_result"]}

    Itinerary:
    {state["itinerary"]}

    Organize the response with clear headings, a day-by-day schedule,
    transportation details, accommodation information, and useful tips.
    Do not invent flight prices or hotel availability.
    """

    response = llm.invoke([
        SystemMessage(
            content="You are a professional travel planning assistant."
        ),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Graph Construction
graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)
graph.add_node("general_chat", general_chat_agent)

# Conditional Entry Point Routing
graph.add_conditional_edges(
    START,
    intent_classifier,
    {
        "travel_plan": "flight_agent",
        "general_chat": "general_chat"
    }
)

# Travel Workflow Edges
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

# Fallback Workflow Edges
graph.add_edge("general_chat", END)

# Database Checkpointer Setup
_conn = psycopg.connect(DATABASE_URL, autocommit=True)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "travel_session_001"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_result": "",
            "hotel_result": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )
    print("\n=== RESPONSE ===\n")
    print(result["messages"][-1].content)