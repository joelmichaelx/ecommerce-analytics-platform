"""
Prometheus Configuration and Metrics Collection
Advanced monitoring setup for E-commerce Analytics Platform
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, start_http_server
import psutil
import redis
import requests
from sqlalchemy import create_engine, text
import json

class EcommerceMetricsCollector:
    """Prometheus metrics collector for E-commerce Analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_connections()
        self.setup_metrics()
    
    def setup_connections(self):
        """Setup connections to various services"""
        try:
            # Database connection
            self.db_engine = create_engine(
                f"postgresql://{self.config['postgres_user']}:{self.config['postgres_password']}@"
                f"{self.config['postgres_host']}:{self.config['postgres_port']}/{self.config['postgres_db']}"
            )
            
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                password=self.config.get('redis_password')
            )
            
            self.logger.info("Monitoring connections established")
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring connections: {str(e)}")
            raise
    
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        try:
            # Create custom registry
            self.registry = CollectorRegistry()
            
            # Business metrics
            self.orders_total = Counter(
                'ecommerce_orders_total',
                'Total number of orders',
                ['status', 'payment_method'],
                registry=self.registry
            )
            
            self.revenue_total = Counter(
                'ecommerce_revenue_total',
                'Total revenue in dollars',
                ['currency', 'category'],
                registry=self.registry
            )
            
            self.customers_total = Gauge(
                'ecommerce_customers_total',
                'Total number of customers',
                ['segment'],
                registry=self.registry
            )
            
            self.products_total = Gauge(
                'ecommerce_products_total',
                'Total number of products',
                ['category', 'status'],
                registry=self.registry
            )
            
            # Performance metrics
            self.order_processing_time = Histogram(
                'ecommerce_order_processing_seconds',
                'Time spent processing orders',
                ['operation'],
                buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=self.registry
            )
            
            self.api_request_duration = Histogram(
                'ecommerce_api_request_duration_seconds',
                'API request duration',
                ['method', 'endpoint', 'status_code'],
                buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=self.registry
            )
            
            self.database_query_duration = Histogram(
                'ecommerce_database_query_duration_seconds',
                'Database query duration',
                ['query_type', 'table'],
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                registry=self.registry
            )
            
            # System metrics
            self.system_cpu_usage = Gauge(
                'ecommerce_system_cpu_usage_percent',
                'System CPU usage percentage',
                registry=self.registry
            )
            
            self.system_memory_usage = Gauge(
                'ecommerce_system_memory_usage_bytes',
                'System memory usage in bytes',
                registry=self.registry
            )
            
            self.system_disk_usage = Gauge(
                'ecommerce_system_disk_usage_bytes',
                'System disk usage in bytes',
                ['device'],
                registry=self.registry
            )
            
            # Data pipeline metrics
            self.data_pipeline_processed_records = Counter(
                'ecommerce_data_pipeline_processed_records_total',
                'Total records processed by data pipeline',
                ['pipeline_stage', 'status'],
                registry=self.registry
            )
            
            self.data_pipeline_processing_time = Histogram(
                'ecommerce_data_pipeline_processing_seconds',
                'Data pipeline processing time',
                ['pipeline_stage'],
                buckets=[1, 5, 10, 30, 60, 300, 600],
                registry=self.registry
            )
            
            # ML model metrics
            self.ml_model_predictions = Counter(
                'ecommerce_ml_model_predictions_total',
                'Total ML model predictions',
                ['model_name', 'prediction_type'],
                registry=self.registry
            )
            
            self.ml_model_accuracy = Gauge(
                'ecommerce_ml_model_accuracy',
                'ML model accuracy',
                ['model_name', 'metric_type'],
                registry=self.registry
            )
            
            # Real-time metrics
            self.realtime_events = Counter(
                'ecommerce_realtime_events_total',
                'Total real-time events processed',
                ['event_type', 'source'],
                registry=self.registry
            )
            
            self.realtime_processing_lag = Gauge(
                'ecommerce_realtime_processing_lag_seconds',
                'Real-time processing lag in seconds',
                ['stream'],
                registry=self.registry
            )
            
            self.logger.info("Prometheus metrics configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup metrics: {str(e)}")
            raise
    
    def collect_business_metrics(self):
        """Collect business-related metrics"""
        try:
            # Orders metrics
            orders_query = """
            SELECT 
                order_status,
                payment_method,
                COUNT(*) as count
            FROM raw_data.orders
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            GROUP BY order_status, payment_method
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(orders_query))
                for row in result:
                    self.orders_total.labels(
                        status=row.order_status,
                        payment_method=row.payment_method
                    ).inc(row.count)
            
            # Revenue metrics
            revenue_query = """
            SELECT 
                currency,
                category,
                SUM(total_amount) as revenue
            FROM processed_data.fact_sales fs
            JOIN processed_data.dim_products dp ON fs.product_key = dp.product_key
            JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
            WHERE dd.full_date >= CURRENT_DATE - INTERVAL '1 day'
            GROUP BY currency, category
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(revenue_query))
                for row in result:
                    self.revenue_total.labels(
                        currency=row.currency or 'USD',
                        category=row.category
                    ).inc(row.revenue)
            
            # Customer metrics
            customers_query = """
            SELECT 
                customer_segment,
                COUNT(*) as count
            FROM analytics.customer_segments
            GROUP BY customer_segment
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(customers_query))
                for row in result:
                    self.customers_total.labels(segment=row.customer_segment).set(row.count)
            
            # Product metrics
            products_query = """
            SELECT 
                category,
                is_active,
                COUNT(*) as count
            FROM raw_data.products
            GROUP BY category, is_active
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(products_query))
                for row in result:
                    self.products_total.labels(
                        category=row.category,
                        status='active' if row.is_active else 'inactive'
                    ).set(row.count)
            
        except Exception as e:
            self.logger.error(f"Failed to collect business metrics: {str(e)}")
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            self.system_disk_usage.labels(device='/').set(disk_usage.used)
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
    
    def collect_data_pipeline_metrics(self):
        """Collect data pipeline metrics"""
        try:
            # Get pipeline processing statistics
            pipeline_query = """
            SELECT 
                'raw_data' as stage,
                COUNT(*) as processed_records,
                'success' as status
            FROM raw_data.orders
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            
            UNION ALL
            
            SELECT 
                'processed_data' as stage,
                COUNT(*) as processed_records,
                'success' as status
            FROM processed_data.fact_sales
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            
            UNION ALL
            
            SELECT 
                'analytics' as stage,
                COUNT(*) as processed_records,
                'success' as status
            FROM analytics.daily_sales_summary
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(pipeline_query))
                for row in result:
                    self.data_pipeline_processed_records.labels(
                        pipeline_stage=row.stage,
                        status=row.status
                    ).inc(row.processed_records)
            
        except Exception as e:
            self.logger.error(f"Failed to collect data pipeline metrics: {str(e)}")
    
    def collect_realtime_metrics(self):
        """Collect real-time streaming metrics"""
        try:
            # Get real-time event counts from Redis
            event_types = ['purchase', 'view', 'add_to_cart', 'remove_from_cart']
            
            for event_type in event_types:
                count = int(self.redis_client.get(f'events:{event_type}') or 0)
                if count > 0:
                    self.realtime_events.labels(
                        event_type=event_type,
                        source='kafka'
                    ).inc(count)
            
            # Get processing lag
            lag_key = 'realtime_processing_lag'
            lag = self.redis_client.get(lag_key)
            if lag:
                self.realtime_processing_lag.labels(stream='sales_events').set(float(lag))
            
        except Exception as e:
            self.logger.error(f"Failed to collect real-time metrics: {str(e)}")
    
    def collect_ml_metrics(self):
        """Collect ML model metrics"""
        try:
            # Get ML model performance from Redis or database
            model_names = ['sales_forecasting', 'customer_segmentation', 'churn_prediction']
            
            for model_name in model_names:
                # Get prediction count
                pred_count = int(self.redis_client.get(f'ml_predictions:{model_name}') or 0)
                if pred_count > 0:
                    self.ml_model_predictions.labels(
                        model_name=model_name,
                        prediction_type='inference'
                    ).inc(pred_count)
                
                # Get model accuracy
                accuracy_key = f'ml_accuracy:{model_name}'
                accuracy = self.redis_client.get(accuracy_key)
                if accuracy:
                    self.ml_model_accuracy.labels(
                        model_name=model_name,
                        metric_type='accuracy'
                    ).set(float(accuracy))
            
        except Exception as e:
            self.logger.error(f"Failed to collect ML metrics: {str(e)}")
    
    def collect_all_metrics(self):
        """Collect all metrics"""
        try:
            self.collect_business_metrics()
            self.collect_system_metrics()
            self.collect_data_pipeline_metrics()
            self.collect_realtime_metrics()
            self.collect_ml_metrics()
            
            self.logger.info("All metrics collected successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {str(e)}")
    
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server"""
        try:
            start_http_server(port, registry=self.registry)
            self.logger.info(f"Prometheus metrics server started on port {port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {str(e)}")
            raise
    
    def run_metrics_collector(self, interval: int = 60):
        """Run metrics collector continuously"""
        try:
            self.logger.info("Starting metrics collector")
            
            while True:
                self.collect_all_metrics()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Metrics collector stopped by user")
        except Exception as e:
            self.logger.error(f"Metrics collector failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    config = {
        'postgres_host': 'localhost',
        'postgres_port': 5432,
        'postgres_db': 'ecommerce_analytics',
        'postgres_user': 'admin',
        'postgres_password': 'password',
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    collector = EcommerceMetricsCollector(config)
    collector.start_metrics_server(port=8000)
    collector.run_metrics_collector(interval=60)
