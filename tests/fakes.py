"""Test double for the Ollama client."""


class FakeBackend:
    """Stands in for ollama.Client. Responses are consumed in order; the last one repeats."""

    def __init__(self, *responses, models=()):
        self.responses = list(responses)
        self.models = models if isinstance(models, Exception) else list(models)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        if kwargs["stream"]:
            pieces = r if isinstance(r, list) else [r]
            return iter({"message": {"content": p}} for p in pieces)
        return {"message": {"content": r}}

    def list(self):
        if isinstance(self.models, Exception):
            raise self.models
        return {"models": [{"model": m} for m in self.models]}
