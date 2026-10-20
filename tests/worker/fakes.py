from PIL import Image

from flux_endpoint.schema import GenerationRequest


class FakeGenerator:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[GenerationRequest, int]] = []
        self._error = error

    @property
    def device_name(self) -> str:
        return "Fake GPU"

    def generate(self, request: GenerationRequest, seed: int) -> Image.Image:
        self.calls.append((request, seed))
        if self._error is not None:
            raise self._error
        return Image.new("RGB", (request.width, request.height), color=(seed % 256, 0, 0))


class FakeStore:
    def __init__(self) -> None:
        self.saved: dict[str, bytes] = {}

    def save(self, name: str, data: bytes) -> str:
        self.saved[name] = data
        return f"https://cdn.example.com/{name}"
