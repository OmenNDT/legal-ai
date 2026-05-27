import logging
import sys
from pathlib import Path

# Bộ logger chung cho toàn pipeline
class Logger:
    _initialized = False

    @classmethod
    def setup(cls, log_dir: Path, name: str = "summarizer"):
        if cls._initialized:
            return logging.getLogger(name)
        log_dir.mkdir(parents=True, exist_ok=True)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        formatter = logging.Formatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        # Ghi ra console
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        # Ghi ra file
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding = "utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        cls._initialized = True
        return logger

    @staticmethod
    def get(name: str = "summarizer"):
        return logging.getLogger(name)
