"""Registry stubs."""


def register(kind: str, name: str, factory) -> None:
    raise NotImplementedError("registry.register is not implemented")


def build(kind: str, name: str, **kwargs):
    raise NotImplementedError("registry.build is not implemented")
