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

**Configure environment variables**

```bash
cp .env.example .env
```

Add your `OPENAI_API_KEY` to `.env`.

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
ollama pull nomic-embed-text
```

**Load SkillsBench skills**

```bash
python src/load_skills.py
```

This downloads SkillsBench v1.1 and generates `data/processed/skills_catalog.json`.

## Running Experiments

**Make the script executable**

```bash
chmod +x run_experiment.sh
```

**Generate queries**

```bash
./run_experiment.sh queries
```

Generates retrieval and usage queries using GPT-4o-mini.

**Generate adversarial skills**

```bash
./run_experiment.sh skills      # Generate adversarial metadata
./run_experiment.sh documents   # Generate adversarial SKILL.md files
```

Generates adversarial skill candidates from training queries using GPT-4o-mini.


**Run retrieval experiments**

```bash
./run_experiment.sh retrieval
```

Generates embeddings using OpenAI (text-embedding-3-small), Nomic (nomic-embed-text), and Sentence-BERT (all-MiniLM-L6-v2) and tests retrieval with adversarial skill candidates.


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
