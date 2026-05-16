from pathlib import Path

class ProjectPath:
    def __init__(self):
        self._root = Path(__file__).resolve().parents[2]

    @property
    def root(self) -> Path:
        return self._root

    def __str__(self) -> str:
        return str(self._root)

    def __repr__(self) -> str:
        return f"ProjectPath(root={self._root!r})"

project_path = ProjectPath()
