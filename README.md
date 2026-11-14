# trojan-agentic-loop
Investigating adversarial attacks on LLM agent tool-calling.

## Setup

**Clone the repository**

```bash
git clone https://github.com/vee-16/trojan-agentic-loop.git
cd trojan-agentic-loop
```

**Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Install and start Ollama**

Install Ollama from https://ollama.com or using Homebrew:

```bash
brew install ollama
ollama serve
```

**Download the models**

In a separate terminal:

```bash
ollama pull gemma4:e2b
ollama pull llama3.1:8b
```

**Load SkillsBench skills**

```bash
python src/load_skills.py
```

This downloads SkillsBench v1.1 and generates the skill catalog.

## Running Experiments

**Make the script executable**

```bash
chmod +x run_experiment.sh
```

**Run the agent**

```bash
./run_experiment.sh agent gemma
./run_experiment.sh agent llama
```

**Run injection experiments**

```bash
./run_experiment.sh inject gemma
./run_experiment.sh inject llama
```

Supported models:
- `gemma` — Gemma 4 E2B
- `llama` — Llama 3.1 8B

The injection experiments test mock adversarial instructions placed at the end or at random positions within the first 10 SkillsBench skill documents. Two payload types are evaluated: evaluation notes and operating guidelines.

Results are saved to `/results/{model}/injection_results.csv`.
