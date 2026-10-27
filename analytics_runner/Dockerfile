# The analytics sandbox image: fixed analysis code plus numpy, scipy and matplotlib, nothing else.
# Build with `make analytics-image` (the only step that needs the network). It runs with
# --network none, a read-only root file system, no capabilities and a non-root user.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/tmp

COPY analytics_runner/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY analytics_runner/runner.py analytics_runner/analyses.py analytics_runner/reference.py /app/
COPY analytics_runner/tests /app/tests

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin runner
USER 10001
WORKDIR /tmp
