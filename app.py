import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final, TypedDict
import yaml

from dotenv import load_dotenv


@dataclass
class SessionAuditLog:
    """Tracks session events and approximate cost."""

    session_id: str
    events: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def log(
        self,
        agent: str,
        action: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        cost = (tokens_in * 0.000015 + tokens_out * 0.00006) / 1000
        self.total_cost_usd += cost
        self.events.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent,
                "action": action,
                "cost_usd": round(cost, 6),
            }
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "events": self.events,
        }

    def save(self, path: Path | None = None) -> Path:
        """Save audit log to JSON file. Returns path used."""
        out = path or Path(__file__).resolve().parent / "audit_logs" / f"{self.session_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out


def format_cost_usd(cost: float) -> str:
    """Format cost as USD string, e.g. $0.00 or $1.23, avoiding scientific notation."""
    if cost >= 0.01:
        return f"${cost:.2f}"
    if cost >= 0.0001:
        return f"${cost:.4f}"
    return f"${cost:.6f}"


def persist_audit_log(audit: SessionAuditLog) -> None:
    path = Path("audit_log.jsonl")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(audit.to_dict()) + "\n")


@dataclass
class AgentHandoff:
    """Structured handoff between supervisor and specialists for typed, auditable routing."""

    from_agent: str
    to_agent: str
    task: str
    context: dict
    priority: str  # "low" | "normal" | "high"
    timestamp: str

    def to_prompt_context(self) -> str:
        return (
            f"HANDOFF FROM {self.from_agent.upper()} TO {self.to_agent.upper()}:\n"
            f"Task: {self.task}\n"
            f"Priority: {self.priority}\n"
            f"Context: {self.context}\n"
            f"Received at: {self.timestamp}"
        )


class MultiAgentState(TypedDict, total=False):
    """State shared across supervisor and specialist nodes."""
    user_request: str        # original user message
    route: str               # "orders" | "billing" | "technical" | "subscription" | "general"
    agent_used: str          # which specialist handled it
    specialist_result: str   # raw output from specialist agent
    final_response: str      # final response returned to the user
    escalated: bool          # whether the case was escalated
    level: str               # e.g. severity or priority level
    handoff: AgentHandoff    # auditable handoff from supervisor to specialist
    audit_log: SessionAuditLog  # session audit log for cost tracking


from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
import random
load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
# Slightly warmer for synthesis—helps responses feel more natural and conversational
llm_friendly = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

# Injection detection at graph entry
INJECTION_PATTERNS: Final[list[str]] = [
    r"ignore (your |all |previous )?instructions",
    r"system prompt.*disabled",
    r"you are now a",
    r"repeat.*system prompt",
    r"jailbreak",
]


def detect_injection(user_input: str) -> bool:
    text = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def guard_request(user_input: str) -> str:
    if detect_injection(user_input):
        return "Hey, I'd love to help, but I can only assist with account and order support. Is there something I can help you with there?"
    return user_input


# Load supervisor prompt from YAML (do not hard-code system prompt in Python)
_SUPERVISOR_YAML_PATH = Path(__file__).resolve().parent / "Prompts" / "supervisor_v1.yaml"
with open(_SUPERVISOR_YAML_PATH, encoding="utf-8") as f:
    _supervisor_prompt = yaml.safe_load(f)
SUPERVISOR_SYSTEM_PROMPT = _supervisor_prompt["system"].strip()

VALID_ROUTES = {"orders", "billing", "technical", "subscription", "general"}


def route_to_specialist(state: MultiAgentState) -> str:
    route_map: dict[str, str] = {
        "orders": "orders_agent_node",
        "billing": "billing_agent_node",
        "technical": "technical_agent_node",
        "subscription": "subscription_agent_node",
        "general": "general_agent_node",
    }
    return route_map.get(state["route"], "general_agent_node")


def _mock_tokens(input_content: str = "", output_content: str = "") -> tuple[int, int]:
    """Mock token counts (no SDK usage). ~4 chars per token."""
    tokens_in = max(50, len(input_content) // 4) if input_content else 100
    tokens_out = max(20, len(output_content) // 4) if output_content else 80
    return (tokens_in, tokens_out)


def supervisor_node(state: MultiAgentState) -> dict:
    """Supervisor node: reads user_request, calls LLM with YAML system prompt, writes normalized route into state."""
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=state["user_request"]),
    ]
    response = llm.invoke(messages)
    tokens_in, tokens_out = _mock_tokens(state["user_request"], response.content or "")
    route = response.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "general"
    if audit := state.get("audit_log"):
        audit.log("supervisor", "route_decision", tokens_in, tokens_out)
    return {"route": route}


@tool
def get_order_status(order_id: str) -> str:
    """ Get order status by order id """
    return f"Order {order_id} is in {random.choice(['pending', 'shipped', 'delivered'])} status"

@tool
def process_return(order_id: str) -> str:
    """ Process return by order id """
    return f"Return {order_id} is processed. Return ID is {random.randint(1000, 9999)}"

@tool
def check_payment_status(order_id: str) -> str:
    """ Check payment status by order id """
    return f"Payment {order_id} is {random.choice(['pending', 'completed', 'failed'])}"

@tool
def check_inventory(product_id: str) -> str:
    """ Check inventory by product id """
    return f"Inventory {product_id} is {random.choice(['in stock', 'out of stock'])}"

@tool
def issue_refund(order_id: str) -> str:
    """ Issue refund by order id """
    return f"Refund for order {order_id} is issued. Refund ID is REF#{random.randint(1000, 9999)}"

@tool
def create_bug_report(issue: str) -> str:
    """ Create bug report by issue """
    return f"Bug report for issue {issue} is created. Bug ID is JIRA#{random.randint(1000, 9999)}"

@tool
def create_feature_request(feature: str) -> str:
    """ Create feature request by feature """
    return f"Feature request for feature {feature} is created. Feature ID is JIRA#{random.randint(1000, 9999)}"

@tool
def upgrade_subscription(customer_id: str) -> str:
    """ Upgrade subscription by customer id """
    return f"Subscription for customer {customer_id} is upgraded. Subscription ID is SUB#{random.randint(1000, 9999)}"

@tool
def get_subscription_status(customer_id: str) -> str:
    """ Get subscription status by customer id """
    return f"Subscription for customer {customer_id} is {random.choice(['active', 'inactive', 'cancelled'])}"

# Specialist tool sets
ORDERS_TOOLS = [get_order_status, process_return, check_inventory]
BILLING_TOOLS = [check_payment_status, issue_refund]
TECHNICAL_TOOLS = [create_bug_report, create_feature_request]
SUBSCRIPTION_TOOLS = [upgrade_subscription, get_subscription_status]
GENERAL_TOOLS = [
    get_order_status, process_return, check_payment_status, check_inventory,
    issue_refund, create_bug_report, create_feature_request,
    upgrade_subscription, get_subscription_status,
]

# Specialist agents (ReAct agents with tool-calling)
_orders_agent = create_react_agent(llm, ORDERS_TOOLS)
_billing_agent = create_react_agent(llm, BILLING_TOOLS)
_technical_agent = create_react_agent(llm, TECHNICAL_TOOLS)
_subscription_agent = create_react_agent(llm, SUBSCRIPTION_TOOLS)
_general_agent = create_react_agent(llm, GENERAL_TOOLS)


def _run_specialist(agent, state: MultiAgentState, agent_name: str) -> dict:
    """Invoke a specialist agent and return state updates."""
    result = agent.invoke({"messages": [HumanMessage(content=state["user_request"])]})
    messages = result.get("messages", [])
    content = messages[-1].content if messages and hasattr(messages[-1], "content") else str(messages[-1])
    tokens_in, tokens_out = _mock_tokens(state["user_request"], content)
    if audit := state.get("audit_log"):
        audit.log(agent_name, "handle_request", tokens_in, tokens_out)
    return {"specialist_result": content, "agent_used": agent_name}


def orders_agent_node(state: MultiAgentState) -> dict:
    """Orders specialist: reads user_request, invokes ReAct agent with order tools."""
    return _run_specialist(_orders_agent, state, "orders_agent")


def billing_agent_node(state: MultiAgentState) -> dict:
    """Billing specialist: reads user_request, invokes ReAct agent with billing tools."""
    return _run_specialist(_billing_agent, state, "billing_agent")


def technical_agent_node(state: MultiAgentState) -> dict:
    """Technical specialist: reads user_request, invokes ReAct agent with technical tools."""
    return _run_specialist(_technical_agent, state, "technical_agent")


def subscription_agent_node(state: MultiAgentState) -> dict:
    """Subscription specialist: reads user_request, invokes ReAct agent with subscription tools."""
    return _run_specialist(_subscription_agent, state, "subscription_agent")


def general_agent_node(state: MultiAgentState) -> dict:
    return _run_specialist(_general_agent, state, "general")


REFLECTION_SYSTEM_PROMPT = """You refine customer support responses to sound friendly and natural, like a real person helping out.

Rules:
- Be warm and conversational—write like you're talking to a friend, not a formal letter
- Use natural phrasing: "Hey, good news!" or "I totally get that—here's what I found" instead of stiff corporate language
- Lead with the answer or outcome; add context after if needed
- Keep it concise but not robotic—a little personality is welcome
- Preserve all factual content (IDs, statuses, next steps)
- Show empathy when things go wrong (e.g., "Sorry to hear that" or "That's frustrating")
- Output ONLY the refined response, no meta-commentary"""


def synthesize_response_node(state: MultiAgentState) -> dict:
    """Synthesize the specialist result into a final user-facing response, with reflection to sharpen it."""
    result = state.get("specialist_result", "")
    agent = state.get("agent_used", "general")
    user_request = state.get("user_request", "")

    if not result:
        return {"final_response": "Oops, I ran into a snag and couldn't quite get that. Could you try again or rephrase your question? I'm here to help!"}

    # Reflection pass: sharpen the response
    messages = [
        SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Original request: {user_request}\n\nSpecialist response to refine:\n{result}"
        ),
    ]
    refined = llm_friendly.invoke(messages)
    final = (refined.content or result).strip()

    tokens_in, tokens_out = _mock_tokens(result, final)
    if audit := state.get("audit_log"):
        audit.log("synthesize_response", "reflection", tokens_in, tokens_out)

    return {"final_response": final}


def build_graph():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("orders_agent_node", orders_agent_node)
    workflow.add_node("billing_agent_node", billing_agent_node)
    workflow.add_node("technical_agent_node", technical_agent_node)
    workflow.add_node("subscription_agent_node", subscription_agent_node)
    workflow.add_node("general_agent_node", general_agent_node)
    workflow.add_node("synthesize_response", synthesize_response_node)

    workflow.set_entry_point("supervisor_node")

    workflow.add_conditional_edges(
        "supervisor_node",
        route_to_specialist,
    )

    for specialist in [
        "orders_agent_node",
        "billing_agent_node",
        "technical_agent_node",
        "subscription_agent_node",
        "general_agent_node",
    ]:
        workflow.add_edge(specialist, "synthesize_response")

    workflow.add_edge("synthesize_response", END)

    return workflow.compile()


def demo_all_agents(user_input: str, audit: SessionAuditLog) -> None:
    """Demonstrate calling every specialized agent with the same user input."""
    safe_text = guard_request(user_input)
    if safe_text != user_input:
        print("Request blocked:", user_input)
        return

    base_state: MultiAgentState = {
        "user_request": safe_text,
        "route": "general",
        "agent_used": "",
        "specialist_result": "",
        "final_response": "",
        "audit_log": audit,
    }

    agents = [
        ("orders_agent_node", orders_agent_node, "Orders"),
        ("billing_agent_node", billing_agent_node, "Billing"),
        ("technical_agent_node", technical_agent_node, "Technical"),
        ("subscription_agent_node", subscription_agent_node, "Subscription"),
        ("general_agent_node", general_agent_node, "General"),
    ]

    print("\n" + "=" * 60)
    print("DEMO: Calling every specialized agent with user input")
    print("=" * 60)
    print("User input:", user_input)
    print("-" * 60)

    for node_name, node_fn, label in agents:
        state = dict(base_state)
        result = node_fn(state)
        specialist_result = result.get("specialist_result", "")
        print(f"\n[{label} Agent]")
        print(specialist_result[:500] + ("..." if len(specialist_result) > 500 else ""))
        print("-" * 60)


def main() -> None:
    """Demonstrate the multi-agent system with example requests."""
    audit = SessionAuditLog(session_id="demo-session")
    graph = build_graph()

    # Demo 1: Call every specialized agent with the same user input
    demo_input = (
        "My order ORD-123 is late and I need a refund. "
        "Also having login issues and want to upgrade my subscription."
    )
    demo_all_agents(demo_input, audit)

    # Demo 2: Normal supervisor routing (single agent per request)
    print("\n" + "=" * 60)
    print("DEMO: Supervisor routing (single agent per request)")
    print("=" * 60)

    for request in [
        "My order ORD-123 is late, can I return it?",
        "I want to upgrade from Basic to Pro. What will it cost?",
        "I was charged twice for my last order.",
        "The app crashes when I open settings.",
    ]:
        safe_text = guard_request(request)
        state: MultiAgentState = {
            "user_request": safe_text,
            "route": "general",
            "agent_used": "",
            "specialist_result": "",
            "final_response": "",
            "audit_log": audit,
        }
        result = graph.invoke(state)
        print("\nRequest:", request)
        print("Route:", result.get("route"), "| Agent used:", result.get("agent_used"))
        print("Final:", (result.get("final_response") or "")[:300] + ("..." if len(str(result.get("final_response", ""))) > 300 else ""))
        print("---")

    print("\nTotal cost (USD):", format_cost_usd(audit.total_cost_usd))
    persist_audit_log(audit)


if __name__ == "__main__":
    main()
