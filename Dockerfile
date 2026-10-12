FROM node:22-bookworm-slim AS viewer
WORKDIR /build/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV DUPLEX_ARENA_WEB_DIR=/app/web/dist
COPY pyproject.toml README.md ./
COPY duplex_arena/ duplex_arena/
COPY workers/ workers/
RUN pip install --no-cache-dir '.[models]'
COPY configs/ configs/
COPY --from=viewer /build/web/dist web/dist/
EXPOSE 8765
CMD ["duplex-arena", "serve", "--host", "0.0.0.0", "--port", "8765"]
