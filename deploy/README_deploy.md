# Hướng dẫn deploy Legal AI lên worker1

## Tổng quan

- **Backend**: `backend/legal_ai_app.py` — Flask app hợp nhất, port **9010**
- **Modules**: String Matching (`/api/string-matching/*`) + Text Summarisation (`/api/summarize`, `/api/extract`, `/api/documents`, `/api/eval`, `/api/auth`)
- **Frontend**: React SPA build sẵn tại `frontend/dist/`, phục vụ qua Flask
- **Portal**: Caddy proxy port **8451** → localhost:9010, service id `legal_ai`

---

## 1. Lần đầu deploy lên worker1

```bash
# Trên worker1 (user hadoop)
cd ~
git clone <repo-url> legal-ai  # hoặc rsync từ local
cd legal-ai

# Tạo virtualenv
python3 -m venv venv
source venv/bin/activate

# Cài dependencies
pip install flask flask-cors waitress psycopg2-binary pyjwt bcrypt
pip install -r backend/text_sumarisation/deploy/requirements.txt
```

## 2. Build frontend (chạy trên máy dev, copy dist lên worker1)

```bash
# Trên máy dev trong thư mục legal-ai/frontend/
npm install
npm run build
# Kết quả: frontend/dist/

# Copy dist lên worker1
rsync -av frontend/dist/ hadoop@100.81.215.111:~/legal-ai/frontend/dist/
```

## 3. Cài systemd service

```bash
# Copy file service
sudo cp ~/legal-ai/deploy/legal-ai.service /etc/systemd/system/

# Reload và enable
sudo systemctl daemon-reload
sudo systemctl enable legal-ai.service
sudo systemctl start legal-ai.service
sudo systemctl status legal-ai.service
```

## 4. Kiểm tra

```bash
curl http://localhost:9010/api/health
# Expected: {"status":"ok","modules":{"string_matching":true,"text_summarisation":true}}
```

## 5. Cập nhật portal trên master

```bash
# Trên master (user trantinnghia)
sudo systemctl restart caddy bdp-portal
# Portal sẽ hiển thị service "Legal AI" thay cho 2 service cũ
```

## 6. Cập nhật sau khi có thay đổi code

```bash
# Trên worker1
cd ~/legal-ai
git pull
# Nếu có thay đổi frontend:
# rsync dist mới từ máy dev
sudo systemctl restart legal-ai.service
```

---

## Cấu trúc port

| Port | Service | Ghi chú |
|------|---------|---------|
| 9010 | Legal AI backend | Flask app hợp nhất |
| 8451 | Legal AI (qua Caddy) | Tailscale HTTPS |
| 8452 | Solar Classifier | Không thay đổi |

## Variables môi trường (.env)

Xem file `.env` ở gốc project. Quan trọng:
- `BDP_LEGAL_DSN` — DSN PostgreSQL cho auth (mặc định dùng worker1:5432/legal_ai)
- `APP_PORT` — port Flask (mặc định 9010)
- `BART_MODEL` — override model BART nếu muốn dùng model khác
