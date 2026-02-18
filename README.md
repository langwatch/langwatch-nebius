# LangWatch x Nebius: Comparing LLM Models with Agent Simulations

This project demonstrates how to **compare different LLM models** for AI agent quality using **agent simulations** as automated evaluations. Built with [Agno](https://github.com/agno-agi/agno) for the agent framework, [Scenario](https://github.com/langwatch/scenario) for simulation-based testing, and [LangWatch](https://langwatch.ai) for observability.

Models are served through [Nebius AI Studio](https://studio.nebius.com/), which provides access to a wide catalog of open-source and commercial models through a single API.

## The Idea

Instead of manually testing each model, we use **agent simulations**: a simulated user interacts with the agent across realistic banking scenarios, and an LLM judge evaluates the quality of responses. This lets you systematically compare how different models handle the same situations.

## Models Compared

| Model | Provider | File |
|-------|----------|------|
| DeepSeek V3.2 | Nebius | `main_support_agent_deepseek.py` |
| GLM-4.7 | Nebius | `main_support_agent_glm.py` |
| MiniMax M2.1 | Nebius | `main_support_agent_minimax.py` |
| GPT-oss 120B | Nebius | `main_support_agent_openai.py` |
| Claude Sonnet 4.5 | Anthropic | `main_support_agent_claude.py` |

## Test Scenarios

Each model is evaluated against the same set of realistic banking scenarios:

1. **Fraud Investigation** - Customer discovers unauthorized transactions, agent must respond with urgency and offer security actions
2. **Escalation Workflow** - Frustrated customer demands a manager, agent must escalate appropriately
3. **Complex Multi-Issue** - Customer has multiple problems (locked account + fees + missing deposit), agent must handle systematically
4. **Urgent Business** - Frozen business account affecting payroll, agent must recognize urgency and act fast
5. **Simple Inquiry** - Basic question that should be answered directly without over-using tools
6. **Spending Analysis** - Customer wants budgeting help, agent should explore account data
7. **Lost Card Replacement** - Lost debit card requiring immediate security measures
8. **Overdraft Fee Dispute** - Customer disputes a fee, agent must show empathy and offer resolution

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Set up environment variables

Create a `.env` file:

```env
NEBIUS_API_KEY=your_nebius_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key      # only needed for Claude tests
OPENAI_API_KEY=your_openai_api_key            # used by the judge/simulator
LANGWATCH_API_KEY=your_langwatch_api_key      # optional, for observability
```

### 3. Run the tests

```bash
# Run all models
uv run pytest tests-demo/ -v

# Run a specific model
uv run pytest tests-demo/test_demo_deepseek.py -v
uv run pytest tests-demo/test_demo_glm.py -v
uv run pytest tests-demo/test_demo_minimax.py -v
uv run pytest tests-demo/test_demo_openai.py -v
uv run pytest tests-demo/test_demo_claude.py -v
```

Results are tracked in [LangWatch](https://langwatch.ai) for side-by-side comparison across models.

## How It Works

```
                    ┌─────────────────┐
                    │  Scenario Test  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌──────────────┐  ┌─────────┐
     │ Simulated  │  │ Bank Support │  │  Judge  │
     │   User     │  │    Agent     │  │  Agent  │
     │ (GPT-4o)   │  │ (Model X)   │  │ (GPT-4o)│
     └────────────┘  └──────┬───────┘  └─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Summary  │  │ Customer │  │   Next   │
        │  Agent   │  │ Explorer │  │ Message  │
        └──────────┘  └──────────┘  └──────────┘
```

The bank support agent is a multi-agent system with specialized sub-agents for conversation summarization, customer data exploration, and response suggestions. Each model variant uses the same architecture - only the LLM is swapped.

## Project Structure

```
├── main_support_agent_deepseek.py   # DeepSeek V3.2 variant
├── main_support_agent_glm.py        # GLM-4.7 variant
├── main_support_agent_minimax.py    # MiniMax M2.1 variant
├── main_support_agent_openai.py     # GPT-oss 120B variant
├── main_support_agent_claude.py     # Claude Sonnet 4.5 variant
├── agent_config.py                  # Shared model config for sub-agents
├── agents/
│   ├── summary_agent.py             # Conversation analysis
│   ├── next_message_agent.py        # Response suggestions
│   └── customer_explorer_agent.py   # Customer data exploration
├── tests-demo/                      # Scenario tests per model
│   ├── test_demo_deepseek.py
│   ├── test_demo_glm.py
│   ├── test_demo_minimax.py
│   ├── test_demo_openai.py
│   └── test_demo_claude.py
└── tests/                           # Base agent tests
```

## Links

- [Scenario](https://github.com/langwatch/scenario) - Agent simulation testing framework
- [LangWatch](https://langwatch.ai) - LLM observability platform
- [Nebius AI Studio](https://studio.nebius.com/) - Model inference platform
- [Agno](https://github.com/agno-agi/agno) - Agent framework
