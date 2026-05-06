from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

def generate_answer(query, context):
    prompt = f"""
    You are a network engineer assistant.

    Context:
    {context}

    Question:
    {query}

    If not found, say "Not in document".
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ── TEST ──────────────────────────────────────
if __name__ == "__main__":
    test_context = """
    BGP (Border Gateway Protocol) is the routing protocol used to exchange routing 
    information between autonomous systems on the internet. It uses TCP port 179.
    
    OSPF uses Dijkstra's algorithm to calculate the shortest path and supports VLSM and CIDR.
    """

    test_queries = [                          # ✅ indented inside __main__
        "What port does BGP use?",
        "What algorithm does OSPF use?",
        "What is the capital of France?",
    ]

    print("🤖 Testing LLM (ChatGoogleGenerativeAI)")
    print("=" * 50)

    for query in test_queries:               # ✅ indented inside __main__
        print(f"\n❓ Query: {query}")
        answer = generate_answer(query, test_context)
        print(f"💬 Answer: {answer.strip()}")
        print("-" * 50)

    print("\n✅ LLM is working!")