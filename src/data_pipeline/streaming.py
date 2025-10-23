"""
Real-time Streaming Analytics
Handles real-time data processing with Apache Kafka and Spark Streaming
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from kafka import KafkaConsumer, KafkaProducer
import redis
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import structlog

class RealTimeStreaming:
    """Real-time streaming analytics processor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.setup_connections()
    
    def setup_connections(self):
        """Setup Kafka and Redis connections"""
        try:
            # Kafka consumer
            self.kafka_consumer = KafkaConsumer(
                bootstrap_servers=[self.config['kafka_bootstrap_servers']],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                group_id=self.config.get('consumer_group', 'analytics_group'),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            # Kafka producer for processed results
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=[self.config['kafka_bootstrap_servers']],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None
            )
            
            # Redis for caching and real-time metrics
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                password=self.config.get('redis_password')
            )
            
            # Spark session for streaming
            self.spark = SparkSession.builder \
                .appName("EcommerceStreamingAnalytics") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            self.logger.info("Streaming connections established")
            
        except Exception as e:
            self.logger.error(f"Failed to setup streaming connections: {str(e)}")
            raise
    
    def process_sales_events(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual sales events for real-time analytics"""
        try:
            # Extract event data
            event_type = message.get('event_type')
            customer_id = message.get('customer_id')
            product_id = message.get('product_id')
            timestamp = datetime.fromisoformat(message.get('timestamp', datetime.now().isoformat()))
            
            # Calculate real-time metrics
            metrics = {
                'event_id': message.get('event_id'),
                'customer_id': customer_id,
                'product_id': product_id,
                'event_type': event_type,
                'timestamp': timestamp.isoformat(),
                'processed_at': datetime.now().isoformat()
            }
            
            # Add event-specific calculations
            if event_type == 'purchase':
                metrics.update({
                    'revenue': message.get('product_price', 0) * message.get('quantity', 1),
                    'is_conversion': True,
                    'conversion_value': message.get('product_price', 0) * message.get('quantity', 1)
                })
            elif event_type == 'view':
                metrics.update({
                    'revenue': 0,
                    'is_conversion': False,
                    'page_view': True
                })
            elif event_type == 'add_to_cart':
                metrics.update({
                    'revenue': 0,
                    'is_conversion': False,
                    'cart_addition': True
                })
            
            # Update real-time counters
            self._update_realtime_counters(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to process sales event: {str(e)}")
            return None
    
    def _update_realtime_counters(self, metrics: Dict[str, Any]):
        """Update real-time counters in Redis"""
        try:
            # Update global counters
            self.redis_client.incr('total_events')
            self.redis_client.incr(f'events:{metrics["event_type"]}')
            
            # Update customer-specific counters
            customer_key = f'customer:{metrics["customer_id"]}'
            self.redis_client.incr(f'{customer_key}:total_events')
            self.redis_client.incr(f'{customer_key}:events:{metrics["event_type"]}')
            
            # Update product-specific counters
            product_key = f'product:{metrics["product_id"]}'
            self.redis_client.incr(f'{product_key}:total_events')
            self.redis_client.incr(f'{product_key}:events:{metrics["event_type"]}')
            
            # Update revenue counters
            if metrics.get('revenue', 0) > 0:
                self.redis_client.incrbyfloat('total_revenue', metrics['revenue'])
                self.redis_client.incrbyfloat(f'{customer_key}:revenue', metrics['revenue'])
                self.redis_client.incrbyfloat(f'{product_key}:revenue', metrics['revenue'])
            
            # Set expiration for customer and product keys (24 hours)
            self.redis_client.expire(customer_key, 86400)
            self.redis_client.expire(product_key, 86400)
            
        except Exception as e:
            self.logger.error(f"Failed to update real-time counters: {str(e)}")
    
    def calculate_realtime_metrics(self) -> Dict[str, Any]:
        """Calculate real-time metrics from Redis counters"""
        try:
            metrics = {}
            
            # Global metrics
            metrics['total_events'] = int(self.redis_client.get('total_events') or 0)
            metrics['total_revenue'] = float(self.redis_client.get('total_revenue') or 0)
            
            # Event type breakdown
            event_types = ['purchase', 'view', 'add_to_cart', 'remove_from_cart']
            metrics['events_by_type'] = {}
            for event_type in event_types:
                count = int(self.redis_client.get(f'events:{event_type}') or 0)
                metrics['events_by_type'][event_type] = count
            
            # Calculate conversion rate
            total_views = metrics['events_by_type'].get('view', 0)
            total_purchases = metrics['events_by_type'].get('purchase', 0)
            if total_views > 0:
                metrics['conversion_rate'] = (total_purchases / total_views) * 100
            else:
                metrics['conversion_rate'] = 0
            
            # Calculate average order value
            if total_purchases > 0:
                metrics['avg_order_value'] = metrics['total_revenue'] / total_purchases
            else:
                metrics['avg_order_value'] = 0
            
            metrics['timestamp'] = datetime.now().isoformat()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate real-time metrics: {str(e)}")
            return {}
    
    def stream_processing_pipeline(self, topics: List[str]):
        """Main streaming processing pipeline"""
        self.logger.info(f"Starting streaming pipeline for topics: {topics}")
        
        try:
            # Subscribe to topics
            self.kafka_consumer.subscribe(topics)
            
            # Process messages
            for message in self.kafka_consumer:
                try:
                    # Process the message
                    processed_metrics = self.process_sales_events(message.value)
                    
                    if processed_metrics:
                        # Send to analytics topic
                        self.kafka_producer.send(
                            topic='analytics_events',
                            key=processed_metrics['customer_id'],
                            value=processed_metrics
                        )
                        
                        # Calculate and send real-time metrics every 100 events
                        if int(self.redis_client.get('total_events') or 0) % 100 == 0:
                            realtime_metrics = self.calculate_realtime_metrics()
                            self.kafka_producer.send(
                                topic='realtime_metrics',
                                key='global',
                                value=realtime_metrics
                            )
                            
                            # Cache metrics in Redis
                            self.redis_client.setex(
                                'realtime_metrics',
                                300,  # 5 minutes TTL
                                json.dumps(realtime_metrics)
                            )
                    
                except Exception as e:
                    self.logger.error(f"Failed to process message: {str(e)}")
                    continue
            
        except KeyboardInterrupt:
            self.logger.info("Streaming pipeline stopped by user")
        except Exception as e:
            self.logger.error(f"Streaming pipeline failed: {str(e)}")
            raise
        finally:
            self.kafka_consumer.close()
            self.kafka_producer.close()
    
    def spark_streaming_analytics(self):
        """Advanced streaming analytics using Spark Streaming"""
        try:
            # Define schema for sales events
            sales_schema = StructType([
                StructField("event_id", StringType(), True),
                StructField("customer_id", StringType(), True),
                StructField("product_id", StringType(), True),
                StructField("event_type", StringType(), True),
                StructField("product_price", DoubleType(), True),
                StructField("quantity", IntegerType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("session_id", StringType(), True),
                StructField("device_type", StringType(), True),
                StructField("location", StringType(), True)
            ])
            
            # Read from Kafka
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.config['kafka_bootstrap_servers']) \
                .option("subscribe", "sales_events") \
                .option("startingOffsets", "latest") \
                .load()
            
            # Parse JSON and apply schema
            sales_df = df.select(
                from_json(col("value").cast("string"), sales_schema).alias("data")
            ).select("data.*")
            
            # Real-time aggregations
            windowed_metrics = sales_df \
                .withWatermark("timestamp", "10 minutes") \
                .groupBy(
                    window(col("timestamp"), "5 minutes"),
                    col("event_type")
                ) \
                .agg(
                    count("*").alias("event_count"),
                    sum(when(col("event_type") == "purchase", col("product_price") * col("quantity")).otherwise(0)).alias("revenue"),
                    countDistinct("customer_id").alias("unique_customers")
                )
            
            # Write to console (in production, write to database or another Kafka topic)
            query = windowed_metrics \
                .writeStream \
                .outputMode("update") \
                .format("console") \
                .option("truncate", False) \
                .start()
            
            query.awaitTermination()
            
        except Exception as e:
            self.logger.error(f"Spark streaming analytics failed: {str(e)}")
            raise
    
    def generate_realtime_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for real-time dashboard"""
        try:
            # Get real-time metrics
            realtime_metrics = self.calculate_realtime_metrics()
            
            # Get top customers by revenue
            top_customers = self._get_top_customers(limit=10)
            
            # Get top products by events
            top_products = self._get_top_products(limit=10)
            
            # Get hourly trends
            hourly_trends = self._get_hourly_trends()
            
            dashboard_data = {
                'realtime_metrics': realtime_metrics,
                'top_customers': top_customers,
                'top_products': top_products,
                'hourly_trends': hourly_trends,
                'generated_at': datetime.now().isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard data: {str(e)}")
            return {}
    
    def _get_top_customers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top customers by revenue"""
        try:
            # Get all customer keys
            customer_keys = self.redis_client.keys('customer:*')
            top_customers = []
            
            for key in customer_keys:
                customer_id = key.decode('utf-8').split(':')[1]
                revenue = float(self.redis_client.get(f'{key.decode("utf-8")}:revenue') or 0)
                total_events = int(self.redis_client.get(f'{key.decode("utf-8")}:total_events') or 0)
                
                if revenue > 0:
                    top_customers.append({
                        'customer_id': customer_id,
                        'revenue': revenue,
                        'total_events': total_events
                    })
            
            # Sort by revenue and return top N
            top_customers.sort(key=lambda x: x['revenue'], reverse=True)
            return top_customers[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get top customers: {str(e)}")
            return []
    
    def _get_top_products(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top products by events"""
        try:
            # Get all product keys
            product_keys = self.redis_client.keys('product:*')
            top_products = []
            
            for key in product_keys:
                product_id = key.decode('utf-8').split(':')[1]
                total_events = int(self.redis_client.get(f'{key.decode("utf-8")}:total_events') or 0)
                revenue = float(self.redis_client.get(f'{key.decode("utf-8")}:revenue') or 0)
                
                if total_events > 0:
                    top_products.append({
                        'product_id': product_id,
                        'total_events': total_events,
                        'revenue': revenue
                    })
            
            # Sort by total events and return top N
            top_products.sort(key=lambda x: x['total_events'], reverse=True)
            return top_products[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get top products: {str(e)}")
            return []
    
    def _get_hourly_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get hourly trends for the last 24 hours"""
        try:
            trends = {
                'revenue': [],
                'events': [],
                'customers': []
            }
            
            # Get hourly data for last 24 hours
            for hour in range(24):
                hour_key = f'hourly:{datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()}'
                
                revenue = float(self.redis_client.get(f'{hour_key}:revenue') or 0)
                events = int(self.redis_client.get(f'{hour_key}:events') or 0)
                customers = int(self.redis_client.get(f'{hour_key}:customers') or 0)
                
                trends['revenue'].append({
                    'hour': hour,
                    'value': revenue
                })
                trends['events'].append({
                    'hour': hour,
                    'value': events
                })
                trends['customers'].append({
                    'hour': hour,
                    'value': customers
                })
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Failed to get hourly trends: {str(e)}")
            return {'revenue': [], 'events': [], 'customers': []}

# Example usage
if __name__ == "__main__":
    config = {
        'kafka_bootstrap_servers': 'localhost:9092',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'consumer_group': 'analytics_group'
    }
    
    streaming = RealTimeStreaming(config)
    
    # Start streaming pipeline
    topics = ['sales_events', 'customer_events', 'product_events']
    streaming.stream_processing_pipeline(topics)
