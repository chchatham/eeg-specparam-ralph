FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir numpy scipy specparam statsmodels plotly fastapi uvicorn

COPY src/ src/
COPY tests/ tests/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
