FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY graphRAG.py .
RUN mkdir -p /app/data/input /app/data/output
EXPOSE 5001
ENV OLLAMA_BASE_URL="http://svc-ollama:11434/v1"
ENV LLM_MODEL="llama3.2:3b"
CMD ["python", "graphRAG.py"]
