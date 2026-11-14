
#!/bin/bash
set -e

TASK=${1:-agent}
MODEL=${2:-gemma}

case "$MODEL" in
    gemma) export MODEL="gemma4:e2b" ;;
    llama) export MODEL="llama3.1:8b" ;;
    *) echo "Unknown model: $MODEL"; exit 1 ;;
esac

case "$TASK" in
    agent) python src/agent.py ;;
    inject) python src/injection_test.py ;;
    *) echo "Usage: ./run.sh [agent|inject] [gemma|llama]"; exit 1 ;;
esac
