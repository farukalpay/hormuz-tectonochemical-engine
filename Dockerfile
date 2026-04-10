FROM python:3.11-slim

WORKDIR /app

ARG INSTALL_TENSORFLOW=true

COPY code ./code
COPY data ./data
COPY paper ./paper
COPY results ./results
COPY assets ./assets
COPY README.md CITATION.cff LICENSE ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r code/requirements.txt \
    && if [ "$INSTALL_TENSORFLOW" = "true" ]; then \
         pip install --no-cache-dir -r code/requirements-tf-linux-windows.txt || true; \
       fi \
    && pip install --no-cache-dir -e "./code[dev,tensorflow]" || pip install --no-cache-dir -e "./code[dev]"

ENV PYTHONPATH=/app/code/src:/app/code

CMD ["/bin/sh", "-lc", "python code/scripts/host_doctor.py && python -m mcp_server.server"]
