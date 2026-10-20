"""LLM provider adapters: Anthropic, Gemini (free tier) and OpenRouter (free models).

Every adapter exposes the same two calls used by the graph:

    generate(system, messages, tier)        -> answer text
    structured(system, prompt, schema, tier) -> pydantic instance, or None if unparseable

`tier` is "answer" for the grounded answer and "light" for the guardrail
classifier and query rewriter. Each tier has an ordered model list; when a
model is rate-limited, overloaded or unavailable the next one is tried. Free
tiers have per-model quotas, so light calls go to the lite models and leave
the stronger models' quota for answers.

Pick a provider with LLM_PROVIDER=anthropic|gemini|openrouter, or let
`get_llm()` take the first one with a key set.
"""

import json
import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

ANSWER_MAX_TOKENS = 8000
LIGHT_MAX_TOKENS = 2000
RETRYABLE_STATUS = {403, 404, 408, 429, 500, 502, 503, 504}


class LLMRefusal(Exception):
    """The provider declined to answer (safety filter)."""


class LLMUnavailable(Exception):
    """Every model in the tier failed."""


def parse_json_output(text, schema):
    """Validate model text against a pydantic schema, tolerating code fences and prose."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return schema.model_validate(json.loads(match.group(0)))
    except (ValueError, TypeError):
        return None


def json_instruction(schema):
    return ("\n\nReply with only a JSON object matching this schema, no other text:\n"
            + json.dumps(schema.model_json_schema()))


class AnthropicAdapter:
    name = "anthropic"
    models = {"answer": ["claude-opus-5-5"], "light": ["claude-opus-5-5"]}

    def __init__(self):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()

    def generate(self, system, messages, tier="answer"):
        try:
            response = self.client.beta.messages.create(
                model=self.models[tier][0],
                max_tokens=16000,
                system=system,
                output_config={"effort": "medium" if tier == "answer" else "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=messages,
            )
        except self.anthropic.APIError as error:
            raise LLMUnavailable(str(error)) from error
        if response.stop_reason == "refusal":
            raise LLMRefusal()
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def structured(self, system, prompt, schema, tier="light"):
        try:
            response = self.client.messages.parse(
                model=self.models[tier][0],
                max_tokens=LIGHT_MAX_TOKENS,
                system=system,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
            )
        except self.anthropic.APIError as error:
            raise LLMUnavailable(str(error)) from error
        if response.stop_reason == "refusal":
            raise LLMRefusal()
        return response.parsed_output


class GeminiAdapter:
    name = "gemini"
    # Free tier: 2.5-pro and 2.5-flash-lite are closed to new keys; quotas are per model.
    models = {
        "answer": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-flash-latest"],
        "light": ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-2.5-flash"],
    }

    def __init__(self):
        from google import genai
        from google.genai import errors, types
        self.types = types
        self.errors = errors
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                timeout=90_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _contents(self, messages):
        return [
            self.types.Content(role="model" if m["role"] == "assistant" else "user",
                               parts=[self.types.Part(text=m["content"])])
            for m in messages
        ]

    def _call(self, tier, contents, config):
        failures = []
        for model in self.models[tier]:
            try:
                response = self.client.models.generate_content(
                    model=model, contents=contents, config=config)
            except self.errors.APIError as error:
                if error.code in RETRYABLE_STATUS:
                    failures.append(f"{model}: {error.code}")
                    continue
                raise LLMUnavailable(f"{model}: {error}") from error
            except httpx.TimeoutException:
                failures.append(f"{model}: timeout")
                continue

            feedback = response.prompt_feedback
            if feedback and feedback.block_reason:
                raise LLMRefusal()
            candidate = response.candidates[0] if response.candidates else None
            if candidate and candidate.finish_reason and candidate.finish_reason.name in (
                    "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
                raise LLMRefusal()
            return response
        raise LLMUnavailable("All Gemini models failed: " + "; ".join(failures))

    def generate(self, system, messages, tier="answer"):
        config = self.types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=ANSWER_MAX_TOKENS,
            temperature=0.1,
        )
        return (self._call(tier, self._contents(messages), config).text or "").strip()

    def structured(self, system, prompt, schema, tier="light"):
        config = self.types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=LIGHT_MAX_TOKENS,
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._call(tier, self._contents([{"role": "user", "content": prompt}]), config)
        if isinstance(response.parsed, schema):
            return response.parsed
        return parse_json_output(response.text, schema)


class OpenRouterAdapter:
    name = "openrouter"
    url = "https://openrouter.ai/api/v1/chat/completions"
    # Checked 2026-10-20: inkling / inkling-small :free return 403 outside OpenRouter's
    # listed agent apps; gemma-4-31b-it:free is often rate-limited upstream, so the
    # small LFM model backs it up.
    models = {
        "answer": ["google/gemma-4-31b-it:free", "liquid/lfm-2.5-2.6b:free"],
        "light": ["liquid/lfm-2.5-2.6b:free", "google/gemma-4-31b-it:free"],
    }

    def __init__(self):
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ["OPENROUTER_APIKEY"]
        self.http = httpx.Client(
            timeout=120,
            headers={"Authorization": f"Bearer {key}", "X-Title": "Shadowfax Contract Assistant"},
        )

    def _call(self, tier, messages, max_tokens, response_format=None):
        failures = []
        for model in self.models[tier]:
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                # Reasoning models spend tokens thinking; keep it out of the reply.
                "reasoning": {"exclude": True},
            }
            if response_format:
                body["response_format"] = response_format
            try:
                response = self.http.post(self.url, json=body)
            except httpx.TimeoutException:
                failures.append(f"{model}: timeout")
                continue
            if response.status_code in RETRYABLE_STATUS:
                failures.append(f"{model}: {response.status_code}")
                continue
            if response.status_code != 200:
                raise LLMUnavailable(f"{model}: HTTP {response.status_code} {response.text[:200]}")

            data = response.json()
            if "error" in data or not data.get("choices"):
                failures.append(f"{model}: {data.get('error', 'no choices')}")
                continue
            choice = data["choices"][0]
            if choice.get("finish_reason") == "content_filter":
                raise LLMRefusal()
            return (choice["message"].get("content") or "").strip()
        raise LLMUnavailable("All OpenRouter models failed: " + "; ".join(failures))

    def generate(self, system, messages, tier="answer"):
        return self._call(tier, [{"role": "system", "content": system}, *messages],
                          ANSWER_MAX_TOKENS)

    def structured(self, system, prompt, schema, tier="light"):
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": True,
                            "schema": schema.model_json_schema()},
        }
        text = self._call(
            tier,
            [{"role": "system", "content": system + json_instruction(schema)},
             {"role": "user", "content": prompt}],
            LIGHT_MAX_TOKENS,
            response_format,
        )
        return parse_json_output(text, schema)


ADAPTERS = {
    "anthropic": (AnthropicAdapter, ("ANTHROPIC_API_KEY",)),
    "gemini": (GeminiAdapter, ("GEMINI_API_KEY",)),
    "openrouter": (OpenRouterAdapter, ("OPENROUTER_API_KEY", "OPENROUTER_APIKEY")),
}


def get_llm(provider=None):
    provider = (provider or os.environ.get("LLM_PROVIDER") or "").lower()
    if provider:
        if provider not in ADAPTERS:
            raise ValueError(f"Unknown LLM provider '{provider}'. Use one of {sorted(ADAPTERS)}.")
        return ADAPTERS[provider][0]()
    for adapter, keys in ADAPTERS.values():
        if any(os.environ.get(k) for k in keys):
            return adapter()
    raise RuntimeError("No LLM key found. Set ANTHROPIC_API_KEY, GEMINI_API_KEY or OPENROUTER_APIKEY.")
