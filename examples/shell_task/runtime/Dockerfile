# Minimal Polar runtime image for the shell-harness task type.
# Stripped from examples/calculator/runtime/Dockerfile: no Node stage, no
# build-essential, no multi-language agent CLIs -- just python3 (to call the
# model + run the inlined grader) plus the coreutils/text tools the curated
# shell tasks exercise (wc, cut, tr, seq, expr, printf, rev, awk).
FROM python:3.12-slim-bookworm

ARG POLAR_SHELL_IMAGE_VERSION=1

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/polar

LABEL io.polar.shell-image-version="${POLAR_SHELL_IMAGE_VERSION}"

# coreutils: wc, cut, tr, seq, printf, expr, echo. util-linux: rev. gawk: awk.
# git: the task_template prepare step runs `git init` in the workspace.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        gawk \
        git \
        sudo \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash polar \
    && echo 'polar ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && mkdir -p /polar/session \
    && chown -R polar:polar /polar/session /home/polar

USER polar

WORKDIR /polar/session
