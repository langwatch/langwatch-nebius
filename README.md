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

## Results

We ran each model through all 8 scenarios, 5 rounds each (40 runs per model). Here's how they compared:

### Cost, Tokens & Latency per Model

| Model | Invokes | Avg Cost/Invoke | Total Cost | Avg Prompt Tok | Avg Compl Tok | Avg Duration |
|-------|---------|-----------------|------------|----------------|---------------|--------------|
| MiniMax-M2.1 | 175 | $0.001022 | $0.18 | 1,248 | 721 | 9,480 ms |
| Claude Opus 4.6 | 157 | $0.019214 | $3.02 | 462 | 676 | 11,953 ms |
| DeepSeek-V3.2 | 206 | $0.000531 | $0.11 | 1,296 | 546 | 14,895 ms |
| GPT-OSS-120B | 162 | $0.000164 | $0.03 | 1,134 | 630 | 3,692 ms |
| GLM-4.7-FP8 | 174 | $0.002034 | $0.35 | 1,260 | 1,020 | 22,369 ms |

### Scenario Pass/Fail (5 rounds x 8 scenarios)

| Model | Passed | Failed | Total | Pass Rate |
|-------|--------|--------|-------|-----------|
| Claude | 26 | 11 | 37 | 70.3% |
| OpenAI | 26 | 13 | 39 | 66.7% |
| DeepSeek | 21 | 18 | 39 | 53.8% |
| GLM | 19 | 20 | 39 | 48.7% |
| MiniMax | 15 | 25 | 40 | 37.5% |

### Key takeaways

- **Claude** has the highest pass rate (70.3%) but is by far the most expensive (~$0.019/invoke, 36x more than GPT-OSS)
- **GPT-OSS-120B** is the fastest (3.7s avg) and cheapest ($0.0002/invoke) but mid-tier quality (66.7%)
- **DeepSeek** is very cheap ($0.0005/invoke) but slowest-ish and moderate quality
- **GLM** is the slowest (22s avg) and generates the most tokens (1,020 avg completion)
- **MiniMax** has the lowest pass rate (37.5%) despite reasonable cost and speed

### Why do invoke counts differ?

The models behave differently during conversations, which changes how many LLM calls they need:

1. **Tool calls create extra invokes** — When a model decides to call a tool (e.g. `explore_customer_account`), it requires another LLM invoke after the tool result to generate the next response. Some models are more tool-happy than others.
2. **DeepSeek (206) is the most tool-heavy** — It frequently called tools multiple times per turn (e.g., calling `explore_customer_account` 2-3 times to "dig deeper"), each one adding an invoke.
3. **Claude (157) is the most efficient** — It tends to handle things in fewer LLM round-trips, making fewer redundant tool calls.
4. **Context loss causes re-asks** — Several models (especially MiniMax and GLM) would "forget" the customer ID mid-conversation and re-ask for it, triggering additional back-and-forth LLM calls that other models avoided.

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
