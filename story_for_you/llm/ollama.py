from story_for_you.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Default LLM provider targeting the Ollama runtime."""

    def __init__(self, model: str = "qwen2.5:7b-instruct", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        """Synchronously generate a response via Ollama."""
        raise NotImplementedError

    def generate_stream(self, prompt: str, system: str = ""):
        """Stream a response via Ollama's API."""
        raise NotImplementedError
