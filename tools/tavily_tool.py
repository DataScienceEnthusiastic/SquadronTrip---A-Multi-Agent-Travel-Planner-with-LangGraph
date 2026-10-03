from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()


client = TavilyClient(
    os.getenv("TAVILY_API_KEY")
)

def tavily_search(query: str) -> dict:
    """
    Search the internet using Tavily.
    """
    try:
        response = client.search(query=query, max_results= 5)

        results = []

        for i,r in enumerate(response["results"], 1):
            title = r.get("title", "Unkown")
            url = r.get("url", "")
            snippet = r.get("content","").strip()
            
            # Keep only the first 300 characters to avoid wall - of - test
            if len(snippet) > 300:
                snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
            
            # format the response
            results.append(
                f" {i}. **{title}**     {url}\n     {snippet}\n"
            )
        
        # If no results found
        if not results:
            return "No results found."
        
        return "\n\n".join(results)   
    except Exception as e:
        return {"error": str(e)}    