# Multi-stage Dockerfile for E-commerce Sales Analytics Platform
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/notebooks

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter

# Expose ports
EXPOSE 8000 8080 8888 3000 9090

# Set default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Change ownership of app directory
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Jupyter stage
FROM base as jupyter

# Install Jupyter dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipykernel

# Expose Jupyter port
EXPOSE 8888

# Set default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# Streamlit stage
FROM base as streamlit

# Install Streamlit dependencies
RUN pip install --no-cache-dir streamlit

# Expose Streamlit port
EXPOSE 8501

# Set default command
CMD ["streamlit", "run", "src/dashboard/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Airflow stage
FROM apache/airflow:2.7.3 as airflow

# Install additional dependencies
USER root
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy DAGs
COPY dags/ /opt/airflow/dags/
COPY plugins/ /opt/airflow/plugins/

# Switch back to airflow user
USER airflow

# Spark stage
FROM openjdk:11-jre-slim as spark

# Install Python and Spark dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install Spark
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Expose Spark ports
EXPOSE 8080 8081 4040 4041

# Set default command
CMD ["spark-submit", "--class", "org.apache.spark.sql.SparkSession", "--master", "local[*]", "src/data_pipeline/spark_warehouse.py"]
