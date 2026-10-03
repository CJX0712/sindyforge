# ---------- 构建阶段 ----------
FROM python:3.13-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY sindyforge ./sindyforge

RUN pip install --no-cache-dir .

# ---------- 运行阶段 ----------
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.authors="晨星"
LABEL org.opencontainers.image.description="SindyForge · 稀疏非线性动力学辨识"

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/sindyforge /usr/local/bin/sindyforge
COPY sindyforge ./sindyforge
COPY examples ./examples

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 非 root 运行
RUN useradd -m -u 1000 sindy && chown -R sindy:sindy /app
USER sindy

ENTRYPOINT ["sindyforge"]
CMD ["run", "--quick"]
