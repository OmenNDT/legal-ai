from fabric import task, Connection
from pathlib import Path

# Cấu hình host worker1 (đã có trong ~/.ssh/config)
WORKER = "worker1"
REMOTE_DIR = "/home/sontn/text_sumarisation"
REMOTE_VENV = f"{REMOTE_DIR}/.venv"
LOCAL_ROOT = Path(__file__).resolve().parents[1]

# Danh sách file/thư mục cần đồng bộ (loại frontend, node_modules, dist...)
RSYNC_INCLUDES = [
    "backend/",
    "data/",
    "deploy/",
]

RSYNC_EXCLUDES = [
    ".venv",
    "__pycache__",
    "*.pyc",
    "node_modules",
    "frontend/dist",
    "outputs/",
    "logs/",
    "backend/cache/",
]

# Build chuỗi --exclude cho rsync
def _exclude_flags():
    return " ".join(f"--exclude='{e}'" for e in RSYNC_EXCLUDES)

# Đảm bảo có connection sẵn
def _conn():
    return Connection(WORKER)

# Đẩy code từ máy local sang worker1
@task
def sync(c):
    flags = _exclude_flags()
    src = str(LOCAL_ROOT) + "/"
    c.local(f"rsync -avz --delete {flags} {src} {WORKER}:{REMOTE_DIR}/")

# Tạo venv và cài deps GPU trên worker1
@task
def setup(c):
    conn = _conn()
    conn.run(f"mkdir -p {REMOTE_DIR}")
    conn.run(f"python3 -m venv {REMOTE_VENV} || true")
    pip = f"{REMOTE_VENV}/bin/pip"
    # Torch CUDA 12.1 (tương thích driver CUDA 13.x của RTX 3090)
    conn.run(f"{pip} install --upgrade pip wheel")
    conn.run(f"{pip} install torch --index-url https://download.pytorch.org/whl/cu121")
    conn.run(f"{pip} install -r {REMOTE_DIR}/deploy/requirements-gpu.txt")

# Kiểm tra GPU bên remote
@task
def gpu(c):
    conn = _conn()
    conn.run("nvidia-smi | head -20")
    conn.run(f"{REMOTE_VENV}/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'")

# Build dataset (chạy 1 lần để cache JSON)
@task
def build_dataset(c):
    conn = _conn()
    conn.run(f"cd {REMOTE_DIR} && PYTHONPATH={REMOTE_DIR} {REMOTE_VENV}/bin/python -c 'from backend.training.run_train import main; main()' --rebuild_dataset --epochs 0 || true", warn = True)

# Bắt đầu fine-tune trên worker1 (nền: dùng nohup)
@task
def train(c, epochs = 3, batch = 2, grad_accum = 8, lora = False):
    conn = _conn()
    lora_flag = "--use_lora" if str(lora).lower() in ("1", "true", "yes") else ""
    cmd = (
        f"cd {REMOTE_DIR} && "
        f"PYTHONPATH={REMOTE_DIR} nohup {REMOTE_VENV}/bin/python -m backend.training.run_train "
        f"--epochs {epochs} --batch_size {batch} --grad_accum {grad_accum} {lora_flag} "
        f"> logs/train.out 2>&1 & echo $! > logs/train.pid"
    )
    conn.run(f"mkdir -p {REMOTE_DIR}/logs")
    conn.run(cmd, pty = False)
    conn.run(f"cat {REMOTE_DIR}/logs/train.pid")

# Theo dõi log training
@task
def tail(c, n = 200):
    conn = _conn()
    conn.run(f"tail -n {n} {REMOTE_DIR}/logs/train.out", warn = True)

# Pull model fine-tune đã lưu về máy local
@task
def pull_model(c, name = "bart-cuad"):
    conn = _conn()
    remote_path = f"{REMOTE_DIR}/outputs/{name}/final"
    local_path = LOCAL_ROOT / "outputs" / name
    local_path.mkdir(parents = True, exist_ok = True)
    c.local(f"rsync -avz {WORKER}:{remote_path}/ {local_path}/")

# Khởi động Flask backend trên worker1 (chạy nền)
@task
def serve(c, port = 9020):
    conn = _conn()
    conn.run(
        f"cd {REMOTE_DIR} && API_PORT={port} PYTHONPATH={REMOTE_DIR} "
        f"nohup {REMOTE_VENV}/bin/python -m backend.app.server > logs/api.out 2>&1 & echo $! > logs/api.pid",
        pty = False,
    )
    conn.run(f"cat {REMOTE_DIR}/logs/api.pid")

# Dừng Flask backend
@task
def stop(c):
    conn = _conn()
    conn.run(f"kill $(cat {REMOTE_DIR}/logs/api.pid) || true", warn = True)
