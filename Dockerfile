FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "numpy>=1.24,<2.1" \
    "scipy>=1.10,<1.15" \
    "specparam>=2.0.0rc0,<3.0" \
    "statsmodels>=0.14,<0.15" \
    "plotly>=5.0,<6.0" \
    "fastapi>=0.100,<1.0" \
    "uvicorn>=0.20,<1.0"

COPY src/ src/
COPY tests/ tests/

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
