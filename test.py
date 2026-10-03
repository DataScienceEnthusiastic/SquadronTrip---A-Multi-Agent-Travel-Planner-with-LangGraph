from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
# result = tavily_search("Weather in Hyderabad")
# print(result)

result = search_flights("7 days trip from India to Japan")
print(result)

