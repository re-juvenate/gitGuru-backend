import getpass
import os
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.llms.fireworks import Fireworks
from langchain_community.llms.ollama import Ollama

config = {}


def set_opts(cfg: dict):
    global config
    config = cfg.copy()


def load_local_llm():
    cfg = config["local"]
    temp = float(cfg["temp"]) if cfg["temp"] is not None else 0.75
    ctx_len = int(cfg["ctx_len"]) if cfg["ctx_len"] is not None else 2048
    max_tokens = (
        int(cfg["max_new_tokens"]) if cfg["max_new_tokens"] is not None else 1024
    )
    n_gpu_layers = int(cfg["gpu_layers"]) if cfg["gpu_layers"] is not None else 24
    llm = LlamaCpp(
        model_path=cfg["model_path"],
        temperature=temp,
        n_gpu_layers=n_gpu_layers,
        n_ctx=ctx_len,
        max_tokens=max_tokens,
    )
    return llm


def load_fireworks_llm():
    if "FIREWORKS_API_KEY" not in os.environ:
        os.environ["FIREWORKS_API_KEY"] = getpass.getpass("Fireworks API Key:")
    cfg = config["fireworks"]

    temp = float(cfg["model_temp"]) if cfg["model_temp"] is not None else 0.75
    max_tokens = (
        int(cfg["model_max_tokens"]) if cfg["model_max_tokens"] is not None else 1024
    )

    # Initialize a Fireworks model
    llm = Fireworks(
        model="accounts/fireworks/models/mistral-7b-instruct-v0p2",
        base_url="https://api.fireworks.ai/inference/v1/completions",
        max_tokens=max_tokens,
        temperature=temp,
    )
    return llm


def load_ollama_llm():
    cfg = config["ollama"]
    temp = float(cfg["temp"]) if cfg["temp"] is not None else 0.75
    ctx_len = int(cfg["ctx_len"]) if cfg["ctx_len"] is not None else 2048
    max_tokens = (
        int(cfg["max_new_tokens"]) if cfg["max_new_tokens"] is not None else 1024
    )
    llm = Ollama(
        model=cfg["model"],
        num_ctx=ctx_len,
        num_predict=max_tokens,
        temperature=temp,
        num_gpu=cfg["num_gpu"],
        format=cfg["format"],
        headers=cfg["headers"],
        mirostat=cfg["mirostat"],
    )
    return llm


def load_ollama2_llm():
    cfg = config["ollama2"]
    temp = float(cfg["temp"]) if cfg["temp"] is not None else 0.75
    ctx_len = int(cfg["ctx_len"]) if cfg["ctx_len"] is not None else 2048
    max_tokens = (
        int(cfg["max_new_tokens"]) if cfg["max_new_tokens"] is not None else 1024
    )
    llm = Ollama(
        model=cfg["model"],
        num_ctx=ctx_len,
        num_predict=max_tokens,
        temperature=temp,
        num_gpu=cfg["num_gpu"],
        format=cfg["format"],
        headers=cfg["headers"],
        mirostat=cfg["mirostat"],
        cache=True,
    )
    return llm


def load_llm(provider="local"):
    if provider == "local":
        return load_local_llm()
    elif provider == "fireworks":
        return load_fireworks_llm()
    elif provider == "ollama":
        return load_ollama_llm()
    elif provider == "ollama2":
        return load_ollama2_llm()


def load_all_llms():
    llms = config["llms"]
    llms = llms if llms is list else [llms]
    loaded = dict.fromkeys(llms)
    for provider in loaded:
        loaded[provider] = load_llm(provider)
    return loaded
