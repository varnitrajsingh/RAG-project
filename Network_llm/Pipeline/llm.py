"""
llm.py

Changes:
- Guard against empty / None context — returns "Not in document." immediately
- Detects low_confidence flag from hybrid.py and prepends a warning to the answer
- Logs context length, confidence level, and token preview before LLM call
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────────────────────
# LLM Init
# ─────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
)

# ─────────────────────────────────────────────────────────────
# Answer Generation  (UPDATED)
# ─────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context: str,
    low_confidence: bool = False,
) -> str:
    """
    Generate an answer from context using the LLM.

    Args:
        query          : The user's question.
        context        : Retrieved context string from the RAG pipeline.
        low_confidence : If True, no chunk passed the score threshold —
                         the answer is prefixed with a confidence warning.

    Returns:
        Answer string.
    """

    print("\n" + "=" * 60)
    print("🤖 LLM ANSWER GENERATION")
    print("=" * 60)

    # ── Guard: empty context ──────────────────────────────────
    if not context or not context.strip():
        print("⚠️  Context is empty — skipping LLM call.")
        print("❌ Returning: 'Not in document.'")
        return "Not in document."

    print(f"📝 Query            : {query}")
    print(f"📏 Context length   : {len(context)} chars")
    print(f"🔍 Context preview  : {context[:200]}...")
    print(f"⚠️  Low confidence  : {low_confidence}")

    # ── Guard: low confidence fallback ───────────────────────
    if low_confidence:
        print("⚠️  Low confidence mode — LLM will answer but result will be prefixed with warning.")

    # ── Build prompt ─────────────────────────────────────────
    prompt = f"""You are a network engineer assistant. Answer ONLY from the context below.
If the answer is not present in the context, respond with exactly: "Not in document."

Context:
{context}

Question:
{query}
"""

    print("\n📤 Sending prompt to LLM...")

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    print(f"📥 LLM raw answer   : {answer[:300]}...")

    # ── Prepend warning if low confidence ────────────────────
    if low_confidence:
        answer = (
            "⚠️ Low confidence — no strongly relevant chunk was found. "
            "The answer below may be inaccurate:\n\n"
            + answer
        )
        print("⚠️  Low confidence prefix added to answer.")

    print("✅ Answer generation complete.")
    return answer


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_context = """
BGP (Border Gateway Protocol) is the routing protocol used to exchange routing
information between autonomous systems on the internet. It uses TCP port 179.

OSPF uses Dijkstra's algorithm to calculate the shortest path and supports VLSM and CIDR.
"""

    test_cases = [
        # (query, context, low_confidence)
        ("What port does BGP use?",        test_context, False),
        ("What algorithm does OSPF use?",  test_context, False),
        ("What is the capital of France?", test_context, False),
        ("What port does BGP use?",        "",           False),   # empty context guard
        ("What port does BGP use?",        test_context, True),    # low confidence path
    ]

    print("\n" + "=" * 80)
    print("🧪 TESTING llm.py")
    print("=" * 80)

    for query, ctx, lc in test_cases:
        print(f"\n❓ Query: {query}")
        print(f"   low_confidence={lc} | context_len={len(ctx)}")
        answer = generate_answer(query, ctx, low_confidence=lc)
        print(f"💬 Final Answer:\n{answer}")
        print("-" * 60)

    print("\n✅ llm.py working!")