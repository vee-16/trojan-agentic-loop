#!/bin/bash
set -e

TASK=${1:-agent}
MODEL=gemma4:e2b

run_injection() {
    local number=$1
    local file="src/attack/injection/injection_test_${number}.py"

    if [[ ! -f "$file" ]]; then
        echo "Injection test $number not found"
        exit 1
    fi

    python "$file"
}

case "$TASK" in
    queries)
        python src/attack/queries/generate_retrieval_queries.py
        python src/attack/queries/generate_usage_queries.py
        ;;
    skills)
        python src/attack/skills/generate_adversarial_skills.py
        ;;
    retrieval)
        python src/attack/skills/retrieval_test.py
        ;;
    agent)
        case "${2:-gemma}" in
            gemma) export MODEL="gemma4:e2b" ;;
            llama) export MODEL="llama3.1:8b" ;;
            *) echo "Unknown model: $2"; exit 1 ;;
        esac
        python src/agent.py
        ;;
    inject)
        if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
            NUMBER=$2
            MODEL_NAME=${3:-gemma}
        else
            NUMBER=all
            MODEL_NAME=${2:-gemma}
        fi

        case "$MODEL_NAME" in
            gemma) export MODEL="gemma4:e2b" ;;
            llama) export MODEL="llama3.1:8b" ;;
            *) echo "Unknown model: $MODEL_NAME"; exit 1 ;;
        esac

        if [[ "$NUMBER" == "all" ]]; then
            shopt -s nullglob
            files=(src/attack/injection/injection_test_*.py)

            if (( ${#files[@]} == 0 )); then
                echo "No injection tests found"
                exit 1
            fi

            for file in "${files[@]}"; do
                echo "Running $file ($MODEL)"
                python "$file"
            done
        else
            run_injection "$NUMBER"
        fi
        ;;
    *)
        echo "Usage: ./run_experiment.sh [queries|skills|embeddings|retrieval|agent|inject] [test_number] [gemma|llama]"
        exit 1
        ;;
esac
