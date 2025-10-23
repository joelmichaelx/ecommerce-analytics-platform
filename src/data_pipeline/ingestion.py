"""
Data Ingestion Module
Handles data ingestion from multiple sources including APIs, databases, and files
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import logging
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine
import psycopg2
from kafka import KafkaProducer
import redis
import time

class DataIngestion:
    """Main data ingestion class for handling multiple data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_connections()
    
    def setup_connections(self):
        """Setup database and message queue connections"""
        try:
            # PostgreSQL connection
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
            
            # Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=[self.config['kafka_bootstrap_servers']],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: x.encode('utf-8') if x else None
            )
            
            self.logger.info("All connections established successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup connections: {str(e)}")
            raise
    
    def ingest_from_api(self, api_config: Dict[str, Any]) -> pd.DataFrame:
        """Ingest data from REST API endpoints"""
        self.logger.info(f"Ingesting data from API: {api_config['url']}")
        
        try:
            headers = api_config.get('headers', {})
            params = api_config.get('params', {})
            
            response = requests.get(
                api_config['url'],
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            # Add metadata
            df['ingestion_timestamp'] = datetime.now()
            df['source'] = api_config.get('source_name', 'api')
            
            self.logger.info(f"Successfully ingested {len(df)} records from API")
            return df
            
        except Exception as e:
            self.logger.error(f"API ingestion failed: {str(e)}")
            raise
    
    def ingest_from_database(self, db_config: Dict[str, Any]) -> pd.DataFrame:
        """Ingest data from external database"""
        self.logger.info(f"Ingesting data from database: {db_config['table']}")
        
        try:
            # Create connection to external database
            external_engine = create_engine(
                f"postgresql://{db_config['user']}:{db_config['password']}@"
                f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            # Execute query
            query = db_config.get('query', f"SELECT * FROM {db_config['table']}")
            df = pd.read_sql(query, external_engine)
            
            # Add metadata
            df['ingestion_timestamp'] = datetime.now()
            df['source'] = db_config.get('source_name', 'database')
            
            self.logger.info(f"Successfully ingested {len(df)} records from database")
            return df
            
        except Exception as e:
            self.logger.error(f"Database ingestion failed: {str(e)}")
            raise
    
    def ingest_from_files(self, file_config: Dict[str, Any]) -> pd.DataFrame:
        """Ingest data from files (CSV, JSON, Parquet)"""
        self.logger.info(f"Ingesting data from file: {file_config['file_path']}")
        
        try:
            file_path = file_config['file_path']
            file_type = file_config.get('file_type', 'csv')
            
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_type == 'excel':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Add metadata
            df['ingestion_timestamp'] = datetime.now()
            df['source'] = file_config.get('source_name', 'file')
            
            self.logger.info(f"Successfully ingested {len(df)} records from file")
            return df
            
        except Exception as e:
            self.logger.error(f"File ingestion failed: {str(e)}")
            raise
    
    def generate_synthetic_data(self, data_type: str, num_records: int) -> pd.DataFrame:
        """Generate synthetic data for testing and development"""
        self.logger.info(f"Generating {num_records} synthetic {data_type} records")
        
        np.random.seed(42)
        
        if data_type == 'sales_events':
            # Generate sales events for real-time streaming
            events = []
            for i in range(num_records):
                event = {
                    'event_id': f'EVENT_{i:08d}',
                    'customer_id': f'CUST_{np.random.randint(1, 1001):06d}',
                    'product_id': f'PROD_{np.random.randint(1, 501):06d}',
                    'order_id': f'ORDER_{np.random.randint(1, 10001):08d}',
                    'event_type': np.random.choice(['purchase', 'view', 'add_to_cart', 'remove_from_cart']),
                    'product_price': round(np.random.uniform(10, 1000), 2),
                    'quantity': np.random.randint(1, 10),
                    'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
                    'session_id': f'SESSION_{np.random.randint(1, 1001):06d}',
                    'device_type': np.random.choice(['mobile', 'desktop', 'tablet']),
                    'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']),
                    'location': np.random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
                    'referrer': np.random.choice(['direct', 'google', 'facebook', 'email', 'other'])
                }
                events.append(event)
            
            return pd.DataFrame(events)
        
        elif data_type == 'customer_events':
            # Generate customer behavior events
            events = []
            for i in range(num_records):
                event = {
                    'event_id': f'CUST_EVENT_{i:08d}',
                    'customer_id': f'CUST_{np.random.randint(1, 1001):06d}',
                    'event_type': np.random.choice(['login', 'logout', 'profile_update', 'password_change']),
                    'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
                    'ip_address': f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                    'user_agent': f"Browser_{np.random.randint(1, 100)}",
                    'success': np.random.choice([True, False], p=[0.95, 0.05])
                }
                events.append(event)
            
            return pd.DataFrame(events)
        
        elif data_type == 'product_events':
            # Generate product-related events
            events = []
            for i in range(num_records):
                event = {
                    'event_id': f'PROD_EVENT_{i:08d}',
                    'product_id': f'PROD_{np.random.randint(1, 501):06d}',
                    'event_type': np.random.choice(['price_change', 'stock_update', 'category_change', 'description_update']),
                    'old_value': str(np.random.uniform(10, 1000)),
                    'new_value': str(np.random.uniform(10, 1000)),
                    'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
                    'updated_by': f'USER_{np.random.randint(1, 101):03d}'
                }
                events.append(event)
            
            return pd.DataFrame(events)
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def stream_to_kafka(self, df: pd.DataFrame, topic: str, key_column: Optional[str] = None):
        """Stream DataFrame records to Kafka topic"""
        self.logger.info(f"Streaming {len(df)} records to Kafka topic: {topic}")
        
        try:
            for index, row in df.iterrows():
                # Convert row to dict
                record = row.to_dict()
                
                # Convert datetime objects to strings
                for key, value in record.items():
                    if isinstance(value, datetime):
                        record[key] = value.isoformat()
                
                # Determine key
                key = None
                if key_column and key_column in record:
                    key = str(record[key_column])
                
                # Send to Kafka
                self.kafka_producer.send(
                    topic=topic,
                    key=key,
                    value=record
                )
            
            # Flush to ensure all messages are sent
            self.kafka_producer.flush()
            self.logger.info(f"Successfully streamed {len(df)} records to {topic}")
            
        except Exception as e:
            self.logger.error(f"Kafka streaming failed: {str(e)}")
            raise
    
    def cache_to_redis(self, key: str, data: Any, ttl: int = 3600):
        """Cache data in Redis"""
        try:
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to JSON
                data_json = data.to_json(orient='records')
                self.redis_client.setex(key, ttl, data_json)
            else:
                self.redis_client.setex(key, ttl, json.dumps(data))
            
            self.logger.info(f"Data cached in Redis with key: {key}")
            
        except Exception as e:
            self.logger.error(f"Redis caching failed: {str(e)}")
            raise
    
    def load_to_warehouse(self, df: pd.DataFrame, table_name: str, schema: str = 'raw_data'):
        """Load DataFrame to data warehouse"""
        self.logger.info(f"Loading {len(df)} records to {schema}.{table_name}")
        
        try:
            df.to_sql(
                table_name,
                self.db_engine,
                schema=schema,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            self.logger.info(f"Successfully loaded data to {schema}.{table_name}")
            
        except Exception as e:
            self.logger.error(f"Data warehouse loading failed: {str(e)}")
            raise
    
    def run_ingestion_pipeline(self, ingestion_config: Dict[str, Any]):
        """Run complete ingestion pipeline"""
        self.logger.info("Starting ingestion pipeline")
        
        try:
            # Extract data from sources
            all_data = []
            
            for source in ingestion_config['sources']:
                source_type = source['type']
                
                if source_type == 'api':
                    df = self.ingest_from_api(source)
                elif source_type == 'database':
                    df = self.ingest_from_database(source)
                elif source_type == 'file':
                    df = self.ingest_from_files(source)
                elif source_type == 'synthetic':
                    df = self.generate_synthetic_data(source['data_type'], source['num_records'])
                else:
                    raise ValueError(f"Unknown source type: {source_type}")
                
                all_data.append(df)
            
            # Combine all data
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Stream to Kafka if configured
                if ingestion_config.get('stream_to_kafka'):
                    for topic_config in ingestion_config['kafka_topics']:
                        topic_df = combined_df[combined_df['source'] == topic_config['source_filter']]
                        if not topic_df.empty:
                            self.stream_to_kafka(
                                topic_df,
                                topic_config['topic'],
                                topic_config.get('key_column')
                            )
                
                # Cache to Redis if configured
                if ingestion_config.get('cache_to_redis'):
                    self.cache_to_redis(
                        ingestion_config['redis_key'],
                        combined_df,
                        ingestion_config.get('redis_ttl', 3600)
                    )
                
                # Load to warehouse if configured
                if ingestion_config.get('load_to_warehouse'):
                    self.load_to_warehouse(
                        combined_df,
                        ingestion_config['warehouse_table'],
                        ingestion_config.get('warehouse_schema', 'raw_data')
                    )
            
            self.logger.info("Ingestion pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Ingestion pipeline failed: {str(e)}")
            raise

# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    config = {
        'postgres_host': 'localhost',
        'postgres_port': 5432,
        'postgres_db': 'ecommerce_analytics',
        'postgres_user': 'admin',
        'postgres_password': 'password',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'kafka_bootstrap_servers': 'localhost:9092'
    }
    
    # Ingestion configuration
    ingestion_config = {
        'sources': [
            {
                'type': 'synthetic',
                'data_type': 'sales_events',
                'num_records': 1000
            },
            {
                'type': 'synthetic',
                'data_type': 'customer_events',
                'num_records': 500
            }
        ],
        'stream_to_kafka': True,
        'kafka_topics': [
            {
                'topic': 'sales_events',
                'source_filter': 'synthetic',
                'key_column': 'customer_id'
            }
        ],
        'cache_to_redis': True,
        'redis_key': 'latest_sales_data',
        'redis_ttl': 3600,
        'load_to_warehouse': True,
        'warehouse_table': 'sales_events',
        'warehouse_schema': 'raw_data'
    }
    
    # Run ingestion
    ingestion = DataIngestion(config)
    ingestion.run_ingestion_pipeline(ingestion_config)
