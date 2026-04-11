ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

ARG INSTALL_TENSORFLOW=true
ARG HTE_TENSORFLOW_DISTRIBUTION=cuda

COPY code ./code
COPY data ./data
COPY paper ./paper
COPY results ./results
COPY README.md CITATION.cff LICENSE ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r code/requirements.txt \
    && if [ "$INSTALL_TENSORFLOW" = "true" ]; then \
         case "$HTE_TENSORFLOW_DISTRIBUTION" in \
           rocm) \
             echo "Using TensorFlow from ROCm base image." ;; \
           cuda|auto) \
             pip install --no-cache-dir -r code/requirements-tf-linux-windows.txt ;; \
           cpu|none) \
             pip install --no-cache-dir "tensorflow-cpu>=2.16.2" ;; \
           *) \
             echo "Unsupported HTE_TENSORFLOW_DISTRIBUTION='$HTE_TENSORFLOW_DISTRIBUTION'" >&2; \
             exit 1 ;; \
         esac; \
       fi \
    && if [ "$HTE_TENSORFLOW_DISTRIBUTION" = "cuda" ] || [ "$HTE_TENSORFLOW_DISTRIBUTION" = "auto" ]; then \
         pip install --no-cache-dir -e "./code[dev,tensorflow]" || pip install --no-cache-dir -e "./code[dev]"; \
       else \
         pip install --no-cache-dir -e "./code[dev]"; \
       fi

ENV PYTHONPATH=/app/code/src:/app/code
ENV HTE_MCP_TRANSPORT=streamable-http
ENV FASTMCP_HOST=0.0.0.0
ENV FASTMCP_PORT=8000
ENV FASTMCP_STREAMABLE_HTTP_PATH=/mcp/hormuz
ENV HTE_GPU_MODE=auto
ENV HTE_REQUIRE_GPU=false
ENV HTE_TENSORFLOW_DISTRIBUTION=auto
ENV HTE_MCP_STATELESS_HTTP=true
ENV HTE_MCP_MAX_CONCURRENT_REQUESTS=6
ENV HTE_OAUTH_STATE_FILE=/app/results/state/oauth_state.json
ENV HTE_OAUTH_ACCESS_TOKEN_TTL_SECONDS=3600
ENV HTE_OAUTH_REFRESH_TOKEN_TTL_SECONDS=2592000
ENV HTE_OAUTH_AUTHORIZATION_CODE_TTL_SECONDS=300
ENV HTE_AUDIT_ENABLED=true
ENV HTE_AUDIT_LOG_RESPONSES=true
ENV HTE_AUDIT_MAX_STRING_LENGTH=20000
ENV HTE_EVIDENCE_MAX_ITEMS=100

CMD ["/bin/sh", "-lc", "python code/scripts/host_doctor.py && python -m mcp_server.server"]
