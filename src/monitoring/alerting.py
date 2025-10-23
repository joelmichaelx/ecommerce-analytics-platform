"""
Advanced Alerting System
Comprehensive alerting for E-commerce Analytics Platform
"""

import logging
import smtplib
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import redis
from sqlalchemy import create_engine, text
import pandas as pd

class EcommerceAlerting:
    """Advanced alerting system for E-commerce Analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_connections()
        self.setup_alert_rules()
    
    def setup_connections(self):
        """Setup connections for alerting"""
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
            
            self.logger.info("Alerting connections established")
            
        except Exception as e:
            self.logger.error(f"Failed to setup alerting connections: {str(e)}")
            raise
    
    def setup_alert_rules(self):
        """Setup alert rules and thresholds"""
        self.alert_rules = {
            'business_metrics': {
                'revenue_drop': {
                    'threshold': 0.2,  # 20% drop
                    'time_window': '1h',
                    'severity': 'high',
                    'enabled': True
                },
                'order_volume_drop': {
                    'threshold': 0.3,  # 30% drop
                    'time_window': '1h',
                    'severity': 'high',
                    'enabled': True
                },
                'conversion_rate_drop': {
                    'threshold': 0.15,  # 15% drop
                    'time_window': '2h',
                    'severity': 'medium',
                    'enabled': True
                },
                'high_value_customer_churn': {
                    'threshold': 5,  # 5 customers
                    'time_window': '24h',
                    'severity': 'critical',
                    'enabled': True
                }
            },
            'system_metrics': {
                'cpu_usage_high': {
                    'threshold': 80,  # 80%
                    'time_window': '5m',
                    'severity': 'medium',
                    'enabled': True
                },
                'memory_usage_high': {
                    'threshold': 90,  # 90%
                    'time_window': '5m',
                    'severity': 'high',
                    'enabled': True
                },
                'disk_usage_high': {
                    'threshold': 85,  # 85%
                    'time_window': '10m',
                    'severity': 'medium',
                    'enabled': True
                },
                'database_connection_failure': {
                    'threshold': 1,  # Any failure
                    'time_window': '1m',
                    'severity': 'critical',
                    'enabled': True
                }
            },
            'data_pipeline': {
                'pipeline_failure': {
                    'threshold': 1,  # Any failure
                    'time_window': '5m',
                    'severity': 'high',
                    'enabled': True
                },
                'processing_delay': {
                    'threshold': 300,  # 5 minutes
                    'time_window': '10m',
                    'severity': 'medium',
                    'enabled': True
                },
                'data_quality_issues': {
                    'threshold': 0.05,  # 5% of records
                    'time_window': '1h',
                    'severity': 'medium',
                    'enabled': True
                }
            },
            'ml_models': {
                'model_accuracy_drop': {
                    'threshold': 0.1,  # 10% drop
                    'time_window': '1h',
                    'severity': 'high',
                    'enabled': True
                },
                'prediction_latency_high': {
                    'threshold': 5,  # 5 seconds
                    'time_window': '5m',
                    'severity': 'medium',
                    'enabled': True
                },
                'model_drift_detected': {
                    'threshold': 0.2,  # 20% drift
                    'time_window': '24h',
                    'severity': 'high',
                    'enabled': True
                }
            }
        }
    
    def check_business_metrics_alerts(self) -> List[Dict[str, Any]]:
        """Check business metrics for alerts"""
        alerts = []
        
        try:
            # Revenue drop alert
            if self.alert_rules['business_metrics']['revenue_drop']['enabled']:
                current_revenue = self.get_current_revenue()
                previous_revenue = self.get_previous_revenue('1h')
                
                if previous_revenue > 0:
                    revenue_drop = (previous_revenue - current_revenue) / previous_revenue
                    threshold = self.alert_rules['business_metrics']['revenue_drop']['threshold']
                    
                    if revenue_drop > threshold:
                        alerts.append({
                            'type': 'revenue_drop',
                            'severity': 'high',
                            'message': f'Revenue dropped by {revenue_drop:.1%} in the last hour',
                            'current_value': current_revenue,
                            'previous_value': previous_revenue,
                            'threshold': threshold,
                            'timestamp': datetime.now()
                        })
            
            # Order volume drop alert
            if self.alert_rules['business_metrics']['order_volume_drop']['enabled']:
                current_orders = self.get_current_orders()
                previous_orders = self.get_previous_orders('1h')
                
                if previous_orders > 0:
                    order_drop = (previous_orders - current_orders) / previous_orders
                    threshold = self.alert_rules['business_metrics']['order_volume_drop']['threshold']
                    
                    if order_drop > threshold:
                        alerts.append({
                            'type': 'order_volume_drop',
                            'severity': 'high',
                            'message': f'Order volume dropped by {order_drop:.1%} in the last hour',
                            'current_value': current_orders,
                            'previous_value': previous_orders,
                            'threshold': threshold,
                            'timestamp': datetime.now()
                        })
            
            # Conversion rate drop alert
            if self.alert_rules['business_metrics']['conversion_rate_drop']['enabled']:
                current_conversion = self.get_current_conversion_rate()
                previous_conversion = self.get_previous_conversion_rate('2h')
                
                if previous_conversion > 0:
                    conversion_drop = (previous_conversion - current_conversion) / previous_conversion
                    threshold = self.alert_rules['business_metrics']['conversion_rate_drop']['threshold']
                    
                    if conversion_drop > threshold:
                        alerts.append({
                            'type': 'conversion_rate_drop',
                            'severity': 'medium',
                            'message': f'Conversion rate dropped by {conversion_drop:.1%} in the last 2 hours',
                            'current_value': current_conversion,
                            'previous_value': previous_conversion,
                            'threshold': threshold,
                            'timestamp': datetime.now()
                        })
            
            # High value customer churn alert
            if self.alert_rules['business_metrics']['high_value_customer_churn']['enabled']:
                churned_high_value = self.get_churned_high_value_customers('24h')
                threshold = self.alert_rules['business_metrics']['high_value_customer_churn']['threshold']
                
                if churned_high_value > threshold:
                    alerts.append({
                        'type': 'high_value_customer_churn',
                        'severity': 'critical',
                        'message': f'{churned_high_value} high-value customers churned in the last 24 hours',
                        'current_value': churned_high_value,
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
            
        except Exception as e:
            self.logger.error(f"Failed to check business metrics alerts: {str(e)}")
        
        return alerts
    
    def check_system_metrics_alerts(self) -> List[Dict[str, Any]]:
        """Check system metrics for alerts"""
        alerts = []
        
        try:
            # CPU usage alert
            if self.alert_rules['system_metrics']['cpu_usage_high']['enabled']:
                cpu_usage = self.get_cpu_usage()
                threshold = self.alert_rules['system_metrics']['cpu_usage_high']['threshold']
                
                if cpu_usage > threshold:
                    alerts.append({
                        'type': 'cpu_usage_high',
                        'severity': 'medium',
                        'message': f'CPU usage is {cpu_usage:.1f}% (threshold: {threshold}%)',
                        'current_value': cpu_usage,
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
            
            # Memory usage alert
            if self.alert_rules['system_metrics']['memory_usage_high']['enabled']:
                memory_usage = self.get_memory_usage()
                threshold = self.alert_rules['system_metrics']['memory_usage_high']['threshold']
                
                if memory_usage > threshold:
                    alerts.append({
                        'type': 'memory_usage_high',
                        'severity': 'high',
                        'message': f'Memory usage is {memory_usage:.1f}% (threshold: {threshold}%)',
                        'current_value': memory_usage,
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
            
            # Disk usage alert
            if self.alert_rules['system_metrics']['disk_usage_high']['enabled']:
                disk_usage = self.get_disk_usage()
                threshold = self.alert_rules['system_metrics']['disk_usage_high']['threshold']
                
                if disk_usage > threshold:
                    alerts.append({
                        'type': 'disk_usage_high',
                        'severity': 'medium',
                        'message': f'Disk usage is {disk_usage:.1f}% (threshold: {threshold}%)',
                        'current_value': disk_usage,
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
            
            # Database connection failure alert
            if self.alert_rules['system_metrics']['database_connection_failure']['enabled']:
                if not self.check_database_connection():
                    alerts.append({
                        'type': 'database_connection_failure',
                        'severity': 'critical',
                        'message': 'Database connection failed',
                        'timestamp': datetime.now()
                    })
            
        except Exception as e:
            self.logger.error(f"Failed to check system metrics alerts: {str(e)}")
        
        return alerts
    
    def check_data_pipeline_alerts(self) -> List[Dict[str, Any]]:
        """Check data pipeline for alerts"""
        alerts = []
        
        try:
            # Pipeline failure alert
            if self.alert_rules['data_pipeline']['pipeline_failure']['enabled']:
                failed_pipelines = self.get_failed_pipelines('5m')
                
                if failed_pipelines:
                    alerts.append({
                        'type': 'pipeline_failure',
                        'severity': 'high',
                        'message': f'Data pipeline failures detected: {", ".join(failed_pipelines)}',
                        'failed_pipelines': failed_pipelines,
                        'timestamp': datetime.now()
                    })
            
            # Processing delay alert
            if self.alert_rules['data_pipeline']['processing_delay']['enabled']:
                processing_delay = self.get_processing_delay()
                threshold = self.alert_rules['data_pipeline']['processing_delay']['threshold']
                
                if processing_delay > threshold:
                    alerts.append({
                        'type': 'processing_delay',
                        'severity': 'medium',
                        'message': f'Data processing delay is {processing_delay} seconds (threshold: {threshold}s)',
                        'current_value': processing_delay,
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
            
            # Data quality issues alert
            if self.alert_rules['data_pipeline']['data_quality_issues']['enabled']:
                quality_issues = self.get_data_quality_issues('1h')
                threshold = self.alert_rules['data_pipeline']['data_quality_issues']['threshold']
                
                if quality_issues > threshold:
                    alerts.append({
                        'type': 'data_quality_issues',
                        'severity': 'medium',
                        'message': f'Data quality issues detected: {quality_issues:.1%} of records (threshold: {threshold:.1%})',
                        'current_value': quality_issues,
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
            
        except Exception as e:
            self.logger.error(f"Failed to check data pipeline alerts: {str(e)}")
        
        return alerts
    
    def check_ml_model_alerts(self) -> List[Dict[str, Any]]:
        """Check ML model metrics for alerts"""
        alerts = []
        
        try:
            # Model accuracy drop alert
            if self.alert_rules['ml_models']['model_accuracy_drop']['enabled']:
                current_accuracy = self.get_current_model_accuracy()
                previous_accuracy = self.get_previous_model_accuracy('1h')
                
                if previous_accuracy > 0:
                    accuracy_drop = (previous_accuracy - current_accuracy) / previous_accuracy
                    threshold = self.alert_rules['ml_models']['model_accuracy_drop']['threshold']
                    
                    if accuracy_drop > threshold:
                        alerts.append({
                            'type': 'model_accuracy_drop',
                            'severity': 'high',
                            'message': f'Model accuracy dropped by {accuracy_drop:.1%} in the last hour',
                            'current_value': current_accuracy,
                            'previous_value': previous_accuracy,
                            'threshold': threshold,
                            'timestamp': datetime.now()
                        })
            
            # Prediction latency alert
            if self.alert_rules['ml_models']['prediction_latency_high']['enabled']:
                prediction_latency = self.get_prediction_latency()
                threshold = self.alert_rules['ml_models']['prediction_latency_high']['threshold']
                
                if prediction_latency > threshold:
                    alerts.append({
                        'type': 'prediction_latency_high',
                        'severity': 'medium',
                        'message': f'Prediction latency is {prediction_latency:.1f}s (threshold: {threshold}s)',
                        'current_value': prediction_latency,
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
            
            # Model drift alert
            if self.alert_rules['ml_models']['model_drift_detected']['enabled']:
                model_drift = self.get_model_drift('24h')
                threshold = self.alert_rules['ml_models']['model_drift_detected']['threshold']
                
                if model_drift > threshold:
                    alerts.append({
                        'type': 'model_drift_detected',
                        'severity': 'high',
                        'message': f'Model drift detected: {model_drift:.1%} (threshold: {threshold:.1%})',
                        'current_value': model_drift,
                        'threshold': threshold,
                        'timestamp': datetime.now()
                    })
            
        except Exception as e:
            self.logger.error(f"Failed to check ML model alerts: {str(e)}")
        
        return alerts
    
    def get_current_revenue(self) -> float:
        """Get current revenue"""
        query = """
        SELECT SUM(total_amount) as revenue
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date >= CURRENT_DATE
        """
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text(query))
            return float(result.fetchone()[0] or 0)
    
    def get_previous_revenue(self, time_window: str) -> float:
        """Get previous revenue for comparison"""
        query = f"""
        SELECT SUM(total_amount) as revenue
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date >= CURRENT_DATE - INTERVAL '{time_window}'
        AND dd.full_date < CURRENT_DATE
        """
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text(query))
            return float(result.fetchone()[0] or 0)
    
    def get_current_orders(self) -> int:
        """Get current order count"""
        query = """
        SELECT COUNT(DISTINCT order_id) as orders
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date >= CURRENT_DATE
        """
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text(query))
            return int(result.fetchone()[0] or 0)
    
    def get_previous_orders(self, time_window: str) -> int:
        """Get previous order count for comparison"""
        query = f"""
        SELECT COUNT(DISTINCT order_id) as orders
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date >= CURRENT_DATE - INTERVAL '{time_window}'
        AND dd.full_date < CURRENT_DATE
        """
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text(query))
            return int(result.fetchone()[0] or 0)
    
    def get_current_conversion_rate(self) -> float:
        """Get current conversion rate"""
        # This would typically come from real-time metrics
        return float(self.redis_client.get('conversion_rate') or 0)
    
    def get_previous_conversion_rate(self, time_window: str) -> float:
        """Get previous conversion rate for comparison"""
        # This would typically be stored in time-series data
        return 0.0  # Placeholder
    
    def get_churned_high_value_customers(self, time_window: str) -> int:
        """Get count of churned high-value customers"""
        query = f"""
        SELECT COUNT(*) as churned_customers
        FROM analytics.customer_segments cs
        WHERE cs.customer_segment = 'Champions'
        AND cs.last_order_date < CURRENT_DATE - INTERVAL '{time_window}'
        """
        
        with self.db_engine.connect() as conn:
            result = conn.execute(text(query))
            return int(result.fetchone()[0] or 0)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        import psutil
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent
    
    def get_disk_usage(self) -> float:
        """Get current disk usage percentage"""
        import psutil
        disk = psutil.disk_usage('/')
        return (disk.used / disk.total) * 100
    
    def check_database_connection(self) -> bool:
        """Check if database connection is working"""
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False
    
    def get_failed_pipelines(self, time_window: str) -> List[str]:
        """Get list of failed pipelines"""
        # This would check Airflow or other pipeline systems
        return []  # Placeholder
    
    def get_processing_delay(self) -> int:
        """Get current processing delay in seconds"""
        # This would check pipeline processing times
        return 0  # Placeholder
    
    def get_data_quality_issues(self, time_window: str) -> float:
        """Get data quality issues percentage"""
        # This would check for data quality issues
        return 0.0  # Placeholder
    
    def get_current_model_accuracy(self) -> float:
        """Get current model accuracy"""
        accuracy_key = 'ml_accuracy:sales_forecasting'
        return float(self.redis_client.get(accuracy_key) or 0)
    
    def get_previous_model_accuracy(self, time_window: str) -> float:
        """Get previous model accuracy for comparison"""
        # This would be stored in time-series data
        return 0.0  # Placeholder
    
    def get_prediction_latency(self) -> float:
        """Get current prediction latency"""
        latency_key = 'ml_latency:sales_forecasting'
        return float(self.redis_client.get(latency_key) or 0)
    
    def get_model_drift(self, time_window: str) -> float:
        """Get model drift percentage"""
        # This would calculate model drift
        return 0.0  # Placeholder
    
    def send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert"""
        try:
            if not self.config.get('email_enabled', False):
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.config['email_from']
            msg['To'] = self.config['email_to']
            msg['Subject'] = f"ALERT: {alert['type'].replace('_', ' ').title()}"
            
            body = f"""
            Alert Type: {alert['type']}
            Severity: {alert['severity'].upper()}
            Message: {alert['message']}
            Timestamp: {alert['timestamp']}
            
            Current Value: {alert.get('current_value', 'N/A')}
            Previous Value: {alert.get('previous_value', 'N/A')}
            Threshold: {alert.get('threshold', 'N/A')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['email_user'], self.config['email_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert['type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
    
    def send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert"""
        try:
            if not self.config.get('slack_enabled', False):
                return
            
            webhook_url = self.config['slack_webhook_url']
            
            payload = {
                "text": f"🚨 *{alert['severity'].upper()} ALERT*",
                "attachments": [
                    {
                        "color": "danger" if alert['severity'] == 'critical' else "warning",
                        "fields": [
                            {"title": "Type", "value": alert['type'], "short": True},
                            {"title": "Severity", "value": alert['severity'], "short": True},
                            {"title": "Message", "value": alert['message'], "short": False},
                            {"title": "Timestamp", "value": str(alert['timestamp']), "short": True}
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent for {alert['type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
    
    def store_alert(self, alert: Dict[str, Any]):
        """Store alert in database"""
        try:
            query = """
            INSERT INTO alerts (alert_type, severity, message, current_value, 
                              previous_value, threshold, timestamp, created_at)
            VALUES (:alert_type, :severity, :message, :current_value, 
                   :previous_value, :threshold, :timestamp, NOW())
            """
            
            with self.db_engine.connect() as conn:
                conn.execute(text(query), {
                    'alert_type': alert['type'],
                    'severity': alert['severity'],
                    'message': alert['message'],
                    'current_value': alert.get('current_value'),
                    'previous_value': alert.get('previous_value'),
                    'threshold': alert.get('threshold'),
                    'timestamp': alert['timestamp']
                })
                conn.commit()
            
            self.logger.info(f"Alert stored in database: {alert['type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to store alert: {str(e)}")
    
    def run_alerting_cycle(self):
        """Run complete alerting cycle"""
        try:
            self.logger.info("Starting alerting cycle")
            
            all_alerts = []
            
            # Check all alert types
            all_alerts.extend(self.check_business_metrics_alerts())
            all_alerts.extend(self.check_system_metrics_alerts())
            all_alerts.extend(self.check_data_pipeline_alerts())
            all_alerts.extend(self.check_ml_model_alerts())
            
            # Process alerts
            for alert in all_alerts:
                # Store alert
                self.store_alert(alert)
                
                # Send notifications based on severity
                if alert['severity'] in ['high', 'critical']:
                    self.send_email_alert(alert)
                    self.send_slack_alert(alert)
                elif alert['severity'] == 'medium':
                    self.send_slack_alert(alert)
            
            self.logger.info(f"Alerting cycle completed. {len(all_alerts)} alerts processed")
            
        except Exception as e:
            self.logger.error(f"Alerting cycle failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    config = {
        'postgres_host': 'localhost',
        'postgres_port': 5432,
        'postgres_db': 'ecommerce_analytics',
        'postgres_user': 'admin',
        'postgres_password': 'password',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'email_enabled': True,
        'email_from': 'alerts@company.com',
        'email_to': 'admin@company.com',
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'email_user': 'alerts@company.com',
        'email_password': 'password',
        'slack_enabled': True,
        'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    }
    
    alerting = EcommerceAlerting(config)
    alerting.run_alerting_cycle()
