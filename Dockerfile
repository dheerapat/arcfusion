FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD . /app

WORKDIR /app
RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "server.py"]