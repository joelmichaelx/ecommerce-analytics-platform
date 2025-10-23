# E-commerce Sales Analytics Platform

A comprehensive, enterprise-grade analytics platform for E-commerce businesses built with modern data engineering and machine learning technologies.

## 🚀 Features

### Core Analytics
- **Real-time Sales Analytics** - Live dashboards with streaming data
- **Customer Segmentation** - RFM analysis and behavioral clustering
- **Product Performance** - Top products, category analysis, and trends
- **Sales Forecasting** - ML-powered predictions using multiple algorithms
- **Geographic Analytics** - Location-based insights and mapping

### Data Engineering
- **Apache Airflow** - Workflow orchestration and scheduling
- **Apache Spark** - Large-scale data processing and analytics
- **Apache Kafka** - Real-time streaming and event processing
- **PostgreSQL** - Data warehouse with optimized schemas
- **Redis** - High-performance caching and real-time metrics

### Machine Learning
- **Sales Forecasting** - Time series models (LSTM, XGBoost, LightGBM)
- **Customer Analytics** - Segmentation, churn prediction, lifetime value
- **Anomaly Detection** - Unusual patterns and fraud detection
- **Recommendation Engine** - Product recommendations and personalization

### Monitoring & Alerting
- **Prometheus** - Metrics collection and monitoring
- **Grafana** - Advanced visualization and dashboards
- **Real-time Alerts** - Business and system metric alerts
- **Data Quality** - Automated data validation and quality checks

### APIs & Integration
- **FastAPI** - High-performance REST API
- **Streamlit** - Interactive web dashboards
- **Jupyter Notebooks** - Data science and experimentation
- **Docker** - Containerized deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Real-time     │    │   Analytics     │
│                 │    │   Streaming     │    │   Layer         │
│ • APIs          │───▶│ • Kafka         │───▶│ • PostgreSQL   │
│ • Databases     │    │ • Redis         │    │ • Spark        │
│ • Files         │    │ • Airflow       │    │ • ML Models    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Applications  │    │   Visualization │
│                 │    │                 │    │                 │
│ • Prometheus    │    │ • FastAPI       │    │ • Streamlit     │
│ • Grafana       │    │ • Airflow       │    │ • Jupyter       │
│ • Alerts        │    │ • ML Pipeline   │    │ • Dashboards    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Data Engineering
- **Apache Airflow** 2.7.3 - Workflow orchestration
- **Apache Spark** 3.5.0 - Distributed data processing
- **Apache Kafka** 7.4.0 - Real-time streaming
- **PostgreSQL** 15 - Data warehouse
- **Redis** 7 - Caching and real-time metrics

### Machine Learning
- **Scikit-learn** - Traditional ML algorithms
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **TensorFlow** - Deep learning models
- **PyTorch** - Neural networks
- **Optuna** - Hyperparameter optimization

### Web & APIs
- **FastAPI** - High-performance API framework
- **Streamlit** - Interactive web applications
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Monitoring
- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **Custom Alerting** - Business and system alerts

### Infrastructure
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **Nginx** - Reverse proxy and load balancing

## 📦 Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- 8GB+ RAM recommended
- 50GB+ disk space

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd ecommerce-sales-analytics-platform
```

2. **Set up environment variables**
```bash
cp config.env.example config.env
# Edit config.env with your settings
```

3. **Start the platform**
```bash
# Development environment
docker-compose up -d

# Production environment
docker-compose -f docker-compose.prod.yml up -d
```

4. **Access the services**
- **API Documentation**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501
- **Airflow**: http://localhost:8080 (admin/admin)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jupyter**: http://localhost:8888
- **Prometheus**: http://localhost:9090

### Manual Setup (Development)

1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up database**
```bash
# Start PostgreSQL
docker run -d --name postgres \
  -e POSTGRES_DB=ecommerce_analytics \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 postgres:15

# Initialize database
psql -h localhost -U admin -d ecommerce_analytics -f sql/init.sql
```

3. **Start Redis**
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

4. **Start Kafka**
```bash
# Start Zookeeper
docker run -d --name zookeeper \
  -p 2181:2181 \
  confluentinc/cp-zookeeper:7.4.0

# Start Kafka
docker run -d --name kafka \
  -p 9092:9092 \
  --link zookeeper \
  confluentinc/cp-kafka:7.4.0
```

5. **Run the application**
```bash
# Start API
python -m uvicorn src.api.main:app --reload

# Start Dashboard
streamlit run src/dashboard/streamlit_app.py

# Start Airflow
airflow webserver --port 8080
airflow scheduler
```

## 📊 Usage

### API Endpoints

#### Analytics
```bash
# Sales summary
GET /analytics/sales/summary?start_date=2024-01-01&end_date=2024-01-31

# Sales trends
GET /analytics/sales/trends?granularity=daily

# Top products
GET /analytics/products/top?metric=revenue&limit=10

# Customer segments
GET /analytics/customers/segments
```

#### Real-time
```bash
# Real-time metrics
GET /realtime/metrics

# Recent events
GET /realtime/events?limit=100
```

#### ML Models
```bash
# Make predictions
POST /ml/predict
{
  "model_name": "sales_forecasting",
  "input_data": {"customer_id": "CUST_001", "product_id": "PROD_001"}
}

# Model status
GET /ml/models/status
```

### Dashboard Usage

1. **Access Streamlit Dashboard**
   - Navigate to http://localhost:8501
   - Use sidebar filters to customize views
   - Explore real-time and historical analytics

2. **Airflow Workflows**
   - Access http://localhost:8080
   - Monitor data pipeline execution
   - Trigger manual runs and view logs

3. **Grafana Monitoring**
   - Access http://localhost:3000
   - View system and business metrics
   - Create custom dashboards

### Data Pipeline

The platform includes automated data pipelines:

1. **Data Ingestion** - Collects data from multiple sources
2. **Data Processing** - Transforms and cleans data
3. **Feature Engineering** - Creates ML features
4. **Model Training** - Trains and validates models
5. **Predictions** - Generates real-time predictions
6. **Monitoring** - Tracks data quality and performance

## 🔧 Configuration

### Environment Variables

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ecommerce_analytics
POSTGRES_USER=admin
POSTGRES_PASSWORD=password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# API
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring
PROMETHEUS_HOST=localhost
PROMETHEUS_PORT=9090
GRAFANA_HOST=localhost
GRAFANA_PORT=3000
```

### Customization

1. **Add new data sources** in `src/data_pipeline/ingestion.py`
2. **Create custom ML models** in `src/ml_models/`
3. **Build custom dashboards** in `src/dashboard/`
4. **Configure alerts** in `src/monitoring/alerting.py`

## 📈 Monitoring

### Metrics
- **Business Metrics**: Revenue, orders, customers, conversion rates
- **System Metrics**: CPU, memory, disk usage, response times
- **Data Quality**: Completeness, accuracy, consistency
- **ML Model Performance**: Accuracy, latency, drift detection

### Alerts
- **Business Alerts**: Revenue drops, customer churn, conversion issues
- **System Alerts**: High resource usage, service failures
- **Data Alerts**: Quality issues, pipeline failures
- **ML Alerts**: Model accuracy drops, prediction latency

### Dashboards
- **Executive Dashboard**: High-level KPIs and trends
- **Operational Dashboard**: Real-time metrics and alerts
- **Technical Dashboard**: System performance and health
- **ML Dashboard**: Model performance and predictions

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Load testing
pytest tests/load/
```

### Test Coverage
```bash
pytest --cov=src tests/
```

## 🚀 Deployment

### Production Deployment

1. **Set up production environment**
```bash
# Create production config
cp config.env.example config.prod.env
# Edit with production values

# Deploy with production compose
docker-compose -f docker-compose.prod.yml up -d
```

2. **Configure monitoring**
```bash
# Set up Prometheus targets
# Configure Grafana dashboards
# Set up alerting rules
```

3. **Scale services**
```bash
# Scale API instances
docker-compose up -d --scale api=3

# Scale workers
docker-compose up -d --scale airflow-scheduler=2
```

### Cloud Deployment

The platform can be deployed on:
- **AWS** - ECS, EKS, RDS, ElastiCache
- **Google Cloud** - GKE, Cloud SQL, Memorystore
- **Azure** - AKS, Azure Database, Redis Cache
- **Kubernetes** - Helm charts available

## 📚 Documentation

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Code Documentation
- **Data Pipeline**: `src/data_pipeline/`
- **ML Models**: `src/ml_models/`
- **API**: `src/api/`
- **Dashboard**: `src/dashboard/`

### Jupyter Notebooks
- **Data Exploration**: `notebooks/exploration/`
- **ML Experiments**: `notebooks/ml/`
- **Tutorials**: `notebooks/tutorials/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Format code
black src/
flake8 src/

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ Core analytics platform
- ✅ Real-time streaming
- ✅ ML models
- ✅ Monitoring and alerting

### Phase 2 (Next)
- 🔄 Advanced ML models
- 🔄 Real-time recommendations
- 🔄 A/B testing framework
- 🔄 Advanced visualizations

### Phase 3 (Future)
- 📋 Multi-tenant architecture
- 📋 Advanced security features
- 📋 Cloud-native deployment
- 📋 AI-powered insights

---

**Built with ❤️ for modern E-commerce analytics**
