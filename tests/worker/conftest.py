import pytest

from .fakes import FakeGenerator


@pytest.fixture
def fake_generator() -> FakeGenerator:
    """Generator that records calls and returns a solid image of the requested size."""
    return FakeGenerator()
