# LangChain Agents

Hands-on experiments and learning notebooks for building LLM applications and agents with **LangChain v1**.

This repository explores the core building blocks behind modern agent systems: model integrations, streaming, batching, tool calling, and LangChain agents.

## What this repo covers

### 1. LangChain Agents

The first notebook introduces LangChain agents using `create_agent`.

It covers:

* Creating an agent with a chat model
* Registering Python functions as tools
* Giving an agent a system prompt
* Invoking an agent with messages
* Letting the model decide when to call a tool

Example:

```python
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"The weather in {city} is sunny"

model = ChatOpenRouter(model="openai/gpt-4o-mini")

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather in New York?"}
    ]
})
```

---

### 2. Model Integrations

The second notebook explores connecting LangChain to different model providers through OpenRouter.

Examples include:

* OpenAI models
* Google Gemini models
* Groq models
* OpenRouter

It also demonstrates common model execution patterns.

#### Invoke

```python
response = model.invoke("How are you?")
print(response.content)
```

#### Streaming

Stream model output as it is generated:

```python
for chunk in model.stream("Why do parrots talk?"):
    print(chunk.text, end="", flush=True)
```

#### Batching

Send multiple independent requests together:

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "Why do airplanes fly?",
    "What is the difference between a crocodile and an alligator?"
])

for response in responses:
    print(response.text)
```

The notebook also demonstrates configuring batch concurrency.

---

### 3. Tool Calling

The third notebook focuses on how LLMs interact with external functions.

The basic flow is:

```text
User request
     ↓
    LLM
     ↓
 Tool call
     ↓
Tool execution
     ↓
 Tool result
     ↓
    LLM
     ↓
Final response
```

A simple tool can be defined with LangChain's `@tool` decorator:

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather information."""
    return f"Weather in {location} is sunny"
```

The model can then be bound to the tool:

```python
model_with_tools = model.bind_tools([get_weather])
```

The notebook walks through executing the returned tool calls and passing the results back to the model.

## Repository Structure

```text
langchain-agents/
├── 1-langchainintro.ipynb
├── 2-modelintegration.ipynb
├── 3-tools.ipynb
├── tests/
│   └── __init__.py
└── .gitignore
```

### `1-langchainintro.ipynb`

Introduces LangChain v1 agents and a basic tool-using agent.

### `2-modelintegration.ipynb`

Explores:

* Model initialization
* OpenRouter integrations
* OpenAI models
* Gemini models
* Groq models
* Streaming
* Batch inference
* Concurrency

### `3-tools.ipynb`

Explains:

* Tool schemas
* Tool definitions
* Tool binding
* Tool calls
* Tool execution
* Tool results
* Completing the model/tool interaction loop

## Requirements

The notebooks are written for **Python 3.12**.

Install the main dependencies:

```bash
pip install -U langchain python-dotenv langchain-openrouter
```

Depending on which integrations you use, you may need additional provider packages.

## Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

Not every notebook requires every key. Use only the credentials required by the model integration you are running.

**Never commit API keys to Git.**

## Running the Notebooks

Clone the repository:

```bash
git clone https://github.com/aadeshadmeasy/langchain-agents.git
cd langchain-agents
```

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -U langchain python-dotenv langchain-openrouter jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Then open the notebooks in this order:

1. `1-langchainintro.ipynb`
2. `2-modelintegration.ipynb`
3. `3-tools.ipynb`

## What You'll Learn

By working through this repository, you will get practical experience with:

* LangChain v1 agent creation
* Chat model initialization
* OpenRouter model routing
* LLM provider integrations
* Streaming responses
* Batch inference
* Concurrency controls
* Structured tool definitions
* LLM tool calling
* Executing tool calls
* Passing tool results back into an agent workflow

## The Agent Loop

At a fundamental level, an agent can be understood as:

```text
                 ┌──────────────┐
                 │     User     │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │      LLM     │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  Tool Call?  │
                 └──────┬───────┘
                        │
                  Yes   │   No
                   │    │
                   ▼    ▼
             ┌────────┐  Final
             │  Tool  │ Response
             └───┬────┘
                 │
                 ▼
            Tool Result
                 │
                 └──────────► LLM
```

This is the foundation for building more complex agentic systems.

## Why This Repository Exists

The goal is to understand what actually happens underneath an agent abstraction instead of treating agents as a black box.

A production agent ultimately needs a few fundamental primitives:

```text
Model
  +
Tools
  +
Context
  +
Execution Loop
  +
State
  =
Agent
```

These notebooks focus on those primitives from the LangChain ecosystem.

## Status

This is an **experimental and educational repository**.

The notebooks are primarily intended for:

* Learning
* Experimentation
* Prototyping
* Understanding agent architectures
* Testing different model integrations

It is not currently intended to be a production-ready agent framework.

## Built by Admeasy

Built by **Aadesh Panwar** as part of experimentation and research into agentic AI systems at **Admeasy AI**.

The work explores how LLMs, tools, model routing, and agent execution can be combined to build autonomous software systems.

## License

No license is currently specified for this repository.
