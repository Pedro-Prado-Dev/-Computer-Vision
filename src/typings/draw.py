import dataclasses


@dataclasses.dataclass(frozen=True)
class Border:
    color: tuple[int, int, int]
    thickness: int


@dataclasses.dataclass(frozen=True)
class Text:
    color: tuple[int, int, int]
    thickness: int
    scale: float
