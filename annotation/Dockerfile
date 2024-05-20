FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt /app

RUN python -m venv venv

ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/venv"

RUN pip install -r requirements.txt

COPY . /app

RUN pip install --no-cache-dir .

CMD [ "streamlit", "run", "src/annotation/Start_Page.py", "--server.baseUrlPath", "/annotation"]
