import json

import time
import yaml
from typing import Any, Optional, List, Mapping


from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

import llmloader

# Constants
config = {}
timestr = lambda: time.strftime("%Y-%m-%d")
with open("settings.yaml", "r") as f:
    config = yaml.safe_load(f)
llmloader.set_opts(config)

to_json_prompt = PromptTemplate.from_template(
    """Parse the given input's structure and ONLY output a JSON based on that structure. Output only the JSON in curly brackets, NOTHING else.
    Input: 
    {input}"""
)
json_corrector_prompt = PromptTemplate.from_template(
    """Correct the given input's structure and ONLY output valid corrected JSON. Output ONLY the JSON in curly brackets. Shut up and do the work.
    Input: 
    {input}"""
)


def llm_to_json(input):
    ollama_llm = llmloader.load_llm(provider="ollama")
    jsoner = to_json_prompt | ollama_llm | StrOutputParser()
    json_corrector = json_corrector_prompt | ollama_llm | StrOutputParser()
    x = jsoner.invoke({"input": input})
    x = json_corrector.invoke({"input": x})
    # x = json.loads(x)
    return x


if __name__ == "__main__":
    a = llm_to_json("""""")
    print(a)
    pass
