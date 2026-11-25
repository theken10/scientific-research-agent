from functools import lru_cache

import httpx

from app.config import settings


@lru_cache(maxsize=1)
def _hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(settings.llm_model)
    model = AutoModelForCausalLM.from_pretrained(settings.llm_model, torch_dtype="auto")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


def _generate_hf(messages: list[dict]) -> str:
    import torch

    tok, model = _hf()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=settings.max_new_tokens, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_openai(messages: list[dict]) -> str:
    # works with vLLM, TGI, or an institute LLM proxy that speaks the OpenAI chat format
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    resp = httpx.post(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        json={"model": settings.llm_model, "messages": messages, "temperature": 0.1,
              "max_tokens": settings.max_new_tokens},
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate(messages: list[dict]) -> str:
    if settings.llm_backend == "openai":
        return _generate_openai(messages)
    return _generate_hf(messages)
