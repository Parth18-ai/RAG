"""
Simple ReAct agent with 2 tools:
  1. web_search  (DuckDuckGo, no API key needed)
  2. calculator  (safe Python math)

Task: "Find today's USD-INR rate and calculate the cost of $500"

Setup (no keys required):
    pip install ollama ddgs
    ollama pull llama3.2
Run:
    python agent.py
"""
import ast
import json
import operator

MODEL = "llama3.2"   # the local Ollama model (the "brain")
MAX_STEPS = 6        # safety limit so the loop can never run forever


# ======================================================
# TOOL 1: web search (a normal Python function)
# ======================================================
def web_search(query):
    # DuckDuckGo search: no API key needed
    from ddgs import DDGS

    results = DDGS().text(query, max_results=3)
    snippets = [f"- {r['title']}: {r['body']}" for r in results]
    return "\n".join(snippets)


# ======================================================
# TOOL 2: calculator (a normal Python function)
# We do NOT use plain eval() because it can run dangerous code.
# Instead we parse the expression and allow only numbers and + - * / **
# ======================================================
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def calculator(expression):
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](walk(node.operand))
        raise ValueError("Only numbers and + - * / ** are allowed")

    tree = ast.parse(expression, mode="eval")
    return str(walk(tree.body))


# The "menu" of tools: name -> function
TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}


# ======================================================
# The prompt: tells the LLM which tools exist and how to ask for them
# ======================================================
SYSTEM_PROMPT = """You are an agent that solves tasks step by step using tools.

Tools:
1. web_search - input: a search query (text). Returns snippets from the web.
2. calculator - input: a math expression like "500 * 83.2". Returns the number.

Reply with ONLY a JSON object in exactly this format:
{"thought": "your reasoning", "action": "web_search" or "calculator" or "final_answer", "input": "tool input, or your final answer"}

Rules:
- Use web_search to find facts you do not know (like today's exchange rate).
- Use calculator for ALL math. Never calculate in your head.
- Do ONE action per reply.
- When you have the final result, use action "final_answer".
"""


# ======================================================
# The LLM call: text in, text out
# format="json" forces the model to reply with valid JSON
# ======================================================
def call_llm(messages):
    import ollama

    response = ollama.chat(model=MODEL, messages=messages, format="json")
    return response["message"]["content"]


# ======================================================
# THE AGENT LOOP (ReAct: Thought -> Action -> Observation)
# ======================================================
def run_agent(task):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step} ---")
        reply = call_llm(messages)

        # 1. Read the LLM's reply (it is just text that should be JSON)
        try:
            data = json.loads(reply)
            thought = data.get("thought", "")
            action = data["action"]
            tool_input = str(data["input"])
        except (json.JSONDecodeError, KeyError):
            print("Bad reply from LLM, asking it to retry...")
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "Invalid format. Reply ONLY with JSON: thought, action, input.",
            })
            continue

        print(f"Thought: {thought}")
        print(f"Action: {action}({tool_input!r})")

        # 2. Final answer? Then we are done.
        if action == "final_answer":
            print(f"\nFINAL ANSWER: {tool_input}")
            return tool_input

        # 3. Otherwise YOUR CODE runs the real tool
        if action not in TOOLS:
            observation = f"Unknown tool '{action}'. Use web_search, calculator or final_answer."
        else:
            try:
                observation = TOOLS[action](tool_input)
            except Exception as error:
                observation = f"Tool error: {error}"

        print(f"Observation: {observation[:300]}")

        # 4. Give the result back to the LLM and loop again
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    print("\nStopped: reached MAX_STEPS without a final answer.")
    return None


if __name__ == "__main__":
    run_agent("Find today's EUR-INR rate and calculate the cost of 500 euros")