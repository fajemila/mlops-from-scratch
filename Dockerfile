# 1. Start with a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install our fast package manager
RUN pip install uv

# 4. Copy our project configuration and install dependencies globally inside the container
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# 5. Copy our FastAPI code and our tracked model
COPY src/api/ ./src/api/
COPY mlruns/ ./mlruns/

# 6. Expose the port FastAPI uses
EXPOSE 8000

# 7. Start the Uvicorn web server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]