FROM python:3.11-slim

WORKDIR /app

ARG INSTALL_TENSORFLOW=true

COPY code ./code
COPY data ./data
COPY paper ./paper
COPY results ./results
COPY README.md CITATION.cff LICENSE ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r code/requirements.txt \
    && if [ "$INSTALL_TENSORFLOW" = "true" ]; then \
         pip install --no-cache-dir -r code/requirements-tf-linux-windows.txt || true; \
       fi \
    && pip install --no-cache-dir -e "./code[dev,tensorflow]" || pip install --no-cache-dir -e "./code[dev]"

ENV PYTHONPATH=/app/code/src:/app/code
ENV HTE_MCP_TRANSPORT=streamable-http
ENV FASTMCP_HOST=0.0.0.0
ENV FASTMCP_PORT=8000
ENV FASTMCP_STREAMABLE_HTTP_PATH=/mcp/hormuz
ENV HTE_MCP_MAX_CONCURRENT_REQUESTS=6
ENV HTE_AUDIT_ENABLED=true
ENV HTE_AUDIT_LOG_RESPONSES=true
ENV HTE_AUDIT_MAX_STRING_LENGTH=20000
ENV HTE_EVIDENCE_MAX_ITEMS=100

CMD ["/bin/sh", "-lc", "python code/scripts/host_doctor.py && python -m mcp_server.server"]
