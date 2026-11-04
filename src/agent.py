from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from .tools import TOOLS

root_agent = LlmAgent(
    name="trojan_agent",
    model=LiteLlm(model="ollama_chat/gemma4:12b"),
    instruction="You are a helpful assistant. Use tools when needed.",
    tools=list(TOOLS.values()),
)