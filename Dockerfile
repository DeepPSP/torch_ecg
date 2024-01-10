FROM python:3.11

ENV PATH=$PATH:/root/.local/bin
RUN pip install pipx && pipx install poetry && poetry config virtualenvs.create false

WORKDIR /root/project

# Copy only dependency-related files for caching
COPY poetry.lock pyproject.toml /root/project/
RUN poetry install --no-root

# Copy actual source code
COPY README.md .
COPY torch_ecg_volta torch_ecg_volta
RUN poetry install --only-root
