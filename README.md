# trojan-agentic-loop
Investigating adversarial attacks on LLM agent tool-calling.

## Setup

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Install and start Ollama**

Install Ollama from https://www.ollama.com or using Homebrew:

```bash
brew install ollama
ollama serve
```

**Download the models**

```bash
ollama pull gemma4:12b
ollama pull deepseek-r1:1.5b 
```

**Run the agent using Google ADK**

From the project root:

```bash
adk web
```

Open the interface http://localhost:8000