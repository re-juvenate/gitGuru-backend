import json
import datetime
import time
import yaml
from typing import Any, Optional, List, Mapping


from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import (
    AgentExecutor,
    load_tools,
    tool,
)

from langchain.agents.output_parsers import XMLAgentOutputParser

import llmloader

# Constants
config = {}
timestr = lambda: time.strftime("%Y-%m-%d")
with open("settings.yaml", "r") as f:
    config = yaml.safe_load(f)
llmloader.set_opts(config)

summarizer_prompt = PromptTemplate.from_template(
    """Summarize the given developers' comments on {repo_info}, preserving the meaning and details
    Commments:
    {input}"""
)

to_json_prompt = PromptTemplate.from_template(
    """Parse the given input's structure and ONLY output a JSON based on that structure. Output only the JSON in curly brackets, NOTHING else.
    Input: 
    {input}"""
)

ollama_llm = llmloader.load_llm(provider="ollama")

def llm_to_json(input):
    x = jsoner.invoke({"input": input})
    x = json.loads(x)
    return x


if __name__ == "__main__":
    pass
