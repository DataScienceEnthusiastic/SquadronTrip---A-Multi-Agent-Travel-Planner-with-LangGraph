from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent
# result = tavily_search("Weather in Hyderabad")
# print(result)

# result = search_flights("7 days trip from India to Japan")
# print(result)

user_input = input("Enter travel request:")
response = run_travel_agent(
    user_input = user_input,
    thread_id = "test_user"
)

print("\n FINAL RESPONSE")
print(response["final_answer"])
