from typing import Optional

class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str, base_url: Optional[str] = None):
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self._provider in ("openai", "compatible"):
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        elif self._provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

        return self._client

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048, temperature: float = 0.1) -> str:
        client = self._get_client()

        if self._provider in ("openai", "compatible"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        if self._provider == "anthropic":
            response = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system or "Bạn là chuyên gia pháp lý Việt Nam.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return ""
