from python:3.12-slim

RUN useradd -ms /bin/bash dev

WORKDIR /project

USER dev

ENV VIRTUAL_ENV=/project/.venv
ENV PATH="/project/.venv/bin:$PATH"
