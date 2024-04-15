FROM python:3.11

ENV POETRY_VERSION=1.3.2

WORKDIR /app

RUN pip install "poetry==$POETRY_VERSION"
COPY poetry.lock pyproject.toml /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

COPY . /app/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_client.py"]