from pathlib import Path
from config.path.get_project_path import project_path

class FrontendPath:
    def __init__(self):
        self._root = project_path.root / "frontend"

    @property
    def root(self) -> Path:
        return self._root

    def __str__(self) -> str:
        return str(self._root)

    def __repr__(self) -> str:
        return f"FrontendPath(root={self._root!r})"

frontend_path = FrontendPath()
