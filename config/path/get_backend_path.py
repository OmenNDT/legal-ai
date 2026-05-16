from pathlib import Path
from config.path.get_project_path import project_path

class BackendPath:
    def __init__(self):
        self._root = project_path.root / "backend"

    @property
    def root(self) -> Path:
        return self._root

    def __str__(self) -> str:
        return str(self._root)

    def __repr__(self) -> str:
        return f"BackendPath(root={self._root!r})"

backend_path = BackendPath()
