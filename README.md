# Agentic Day 4 - Multi-Agent

Multi-agent customer support system built with LangChain and LangGraph.

---

## Overview

This project implements a **supervisor–specialist** multi-agent architecture for customer support. A supervisor classifies incoming requests and routes them to specialized agents (orders, billing, technical, subscription, or general). Each specialist has domain-specific tools and uses a ReAct-style agent to handle the request. A final synthesis step refines the response for a friendly, natural tone.

---

## How It Works

```
User Request → [Guard] → Supervisor (classify) → Specialist Agent (ReAct + tools) → Synthesize → Final Response
```

1. **Guard** — Validates input and blocks prompt-injection attempts before processing.
2. **Supervisor** — Uses an LLM with a YAML-defined system prompt to classify the request into one of: `orders`, `billing`, `technical`, `subscription`, or `general`.
3. **Specialist routing** — A conditional edge routes to the appropriate specialist node based on the supervisor’s classification.
4. **Specialist agents** — Each specialist is a ReAct agent with domain-specific tools (e.g., `get_order_status`, `issue_refund`, `create_bug_report`). The agent can call tools and reason over the results.
5. **Synthesis** — A reflection pass refines the specialist’s raw output into a warm, conversational response before returning to the user.
6. **Audit** — Session events and approximate token costs are logged for each step.

---

## Architecture

| Component        | Role                                                                 |
|-----------------|----------------------------------------------------------------------|
| **Supervisor**  | Classifies requests; routes to the correct specialist                |
| **Orders**      | Order status, returns, inventory                                     |
| **Billing**     | Payment status, refunds                                              |
| **Technical**   | Bug reports, feature requests                                        |
| **Subscription**| Upgrades, status, plan changes                                       |
| **General**     | Fallback; has access to all tools                                    |

---

## Project Structure

```
├── app.py                 # Main application: graph, agents, tools, demos
├── Prompts/
│   └── supervisor_v1.yaml # Supervisor system prompt (YAML)
├── requirements.txt
├── .env                   # API keys (do not commit)
└── audit_log.jsonl        # Session audit logs (appended at runtime)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_key_here
```

**Do not commit `.env` to version control.**

---

## Run

```bash
python app.py
```

The demo runs two flows:

1. **All-agents demo** — Sends one multi-topic request to every specialist to show each agent’s response.
2. **Supervisor routing demo** — Sends several requests through the full pipeline (supervisor → specialist → synthesis) to show routing and final responses.

---

## Tools by Specialist

| Specialist   | Tools                                                                 |
|-------------|-----------------------------------------------------------------------|
| Orders      | `get_order_status`, `process_return`, `check_inventory`               |
| Billing     | `check_payment_status`, `issue_refund`                                |
| Technical   | `create_bug_report`, `create_feature_request`                         |
| Subscription| `upgrade_subscription`, `get_subscription_status`                     |
| General     | All of the above                                                      |
