import json
from langchain_community.tools import BraveSearch
from rich import print
from dotenv import load_dotenv

load_dotenv()

search = BraveSearch()

print(json.loads(search.run("what is agentic ai")))
