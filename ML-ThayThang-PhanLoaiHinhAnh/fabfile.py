import subprocess
from fabric import Connection, task

REMOTE = "sontn@100.81.215.111"
REMOTE_DIR = "~/ML-ThayThang-PhanLoaiHinhAnh"

@task
def run(c):
    subprocess.run(
        f"rsync -avz --exclude '__pycache__' . {REMOTE}:{REMOTE_DIR}/",
        shell=True, check=True
    )
    Connection(REMOTE).run(f"cd {REMOTE_DIR} && python3 main.py")
    subprocess.run(
        f"rsync -avz {REMOTE}:{REMOTE_DIR}/models/ ./models/",
        shell = True, check = True
    )