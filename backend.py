import os
import certifi
import dotenv
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict , Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph , END, START

from langchain_core.messages import(
    HumanMessage,
    AIMessage,
    SystemMessage,
    AnyMessage
)

from langchain_groq import ChatGroq
from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is not set in .env file")

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    
    return database_url

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env file")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)

class TravelState(TypedDict):
    messages : Annotated[list[AnyMessage], operator.add]
    user_query : str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# Flight Agent 
def flight_agent(state: TravelState):

    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content=f"Flight results are fetched"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# Hotel Agent 
def hotel_agent(state: TravelState):

    query = f"Best hotels for {state['user_query']}"
    hotel_data = tavily_search(query)

    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(
                content=f"Hotel results are fetched"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# Itinerary Agent 
def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a detailed and comprehensive travel itinerary based on the user's request and the following data:

    User Query: {state['user_query']}
    Flight Results: {state['flight_results']}
    Hotel Results: {state['hotel_results']}

    Instructions:
    1. Use the flight results to determine the best travel dates and times.
    2. Use the hotel results to suggest suitable accommodation options.
    3. Create a day-by-day itinerary that includes:
        - Daily activities and attractions
        - Meal recommendations
        - Transportation details
        - Estimated costs
    4. Present the itinerary in a clear, organized, and visually appealing format.
    5. Ensure the itinerary is tailored to the user's query and the provided data.

    Make sure the itinerary is detailed, comprehensive, and easy to follow.
    
    """
    response = llm.invoke(
        [
            SystemMessage(content="You are expert travel planner "),
            HumanMessage(content= prompt)
        ]
    )

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    
# Final Agent 
def final_agent(state: TravelState):

    final_prompt = f"""
    Generate the final travel response for the user 
    User Request : {state['user_query']}
    Flights : {state['flight_results']}
    Hotels : {state['hotel_results']}
    Itinerary : {state['itinerary']}

    Instructions:
    1. Use the flight results to determine the best travel dates and times.
    2. Use the hotel results to suggest suitable accommodation options.
    3. Create a day-by-day itinerary that includes:
        - Daily activities and attractions
        - Meal recommendations
        - Transportation details
        - Estimated costs
    4. Present the itinerary in a clear, organized, and visually appealing format.
    5. Ensure the itinerary is tailored to the user's query and the provided data.

    Make sure the itinerary is detailed, comprehensive, and easy to follow.

    
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are Professional AI travel booking assistant "),
            HumanMessage(content= final_prompt)
        ]
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# Building a Graph

graph = StateGraph(TravelState)

# Adding nodes
graph.add_node("flight_agent" , flight_agent)
graph.add_node("hotel_agent" , hotel_agent)
graph.add_node("itinerary_agent" , itinerary_agent)
graph.add_node("final_agent" , final_agent)

# Adding edges
graph.add_edge(START , "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

## PostgreSQL Checkpointer 
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    conninfo=DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(conn=_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer = checkpointer)


# Function for FASTAPI

def run_travel_agent(user_input:str, thread_id:str | None = None):

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"
    
    config={
        "configurable": {
            "thread_id":thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages" : [HumanMessage(content=user_input)],
            "user_query" : user_input,
            "flight_results" : "",
            "hotel_results" : "",
            "itinerary" : "",
            "llm_calls" : 0,
        },
        config=config
    )
    final_answer = result["messages"][-1].content
    
    return {
        "thread_id" : thread_id,
        "final_answer" : final_answer,
        "llm_calls" : result.get("llm_calls",0),
        "flight_results" : result.get("flight_results",""),
        "hotel_results" : result.get("hotel_results",""),
        "itinerary" : result.get("itinerary","")
    }

    
    