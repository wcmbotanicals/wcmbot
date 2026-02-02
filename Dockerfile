FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip
RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra gpu --no-install-project

RUN .venv/bin/python - << 'EOF'
from rembg import new_session
new_session("isnet-general-use")
new_session("birefnet-dis")
print("Rembg models downloaded successfully")
EOF

COPY . .
RUN uv pip install -e .

ENTRYPOINT ["/app/.venv/bin/python", "-u", "app.py"]
CMD ["--ai-seg", "--accessible"]
