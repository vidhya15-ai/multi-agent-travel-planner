import os
from typing import TypedDict, Annotated
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

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

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
    print("\n=== FINAL TRAVEL PLAN ===\n")
    print(result["messages"][-1].content)