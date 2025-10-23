"""
Pydantic Models for API
Data models for E-commerce Analytics API
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
from enum import Enum

# Enums
class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(str, Enum):
    PURCHASE = "purchase"
    VIEW = "view"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    LOGIN = "login"
    LOGOUT = "logout"

class CustomerSegment(str, Enum):
    CHAMPIONS = "Champions"
    LOYAL_CUSTOMERS = "Loyal Customers"
    NEW_CUSTOMERS = "New Customers"
    POTENTIAL_LOYALISTS = "Potential Loyalists"
    PROMISING = "Promising"
    NEED_ATTENTION = "Need Attention"
    ABOUT_TO_SLEEP = "About to Sleep"
    AT_RISK = "At Risk"
    CANNOT_LOSE_THEM = "Cannot Lose Them"

class PipelineStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SCHEDULED = "scheduled"
    PAUSED = "paused"

class ModelStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    TRAINING = "training"

# Base models
class BaseResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Analytics models
class SalesSummary(BaseModel):
    total_orders: int
    total_revenue: float
    avg_order_value: float
    unique_customers: int
    total_profit: float
    avg_margin: float
    period: Dict[str, Any]

class SalesTrend(BaseModel):
    period: str
    orders: int
    revenue: float
    customers: int

class SalesTrendsResponse(BaseResponse):
    trends: List[SalesTrend]

class ProductPerformance(BaseModel):
    product_name: str
    category: str
    brand: str
    metric_value: float
    total_revenue: float
    total_quantity: int
    total_orders: int

class TopProductsResponse(BaseResponse):
    top_products: List[ProductPerformance]

class CustomerSegment(BaseModel):
    segment: str
    customer_count: int
    avg_spent: float
    avg_orders: float
    avg_order_value: float

class CustomerSegmentsResponse(BaseResponse):
    customer_segments: List[CustomerSegment]

# Real-time models
class RealtimeMetrics(BaseModel):
    total_events: int
    total_revenue: float
    conversion_rate: float
    avg_order_value: float
    events_by_type: Dict[str, int]

class Event(BaseModel):
    event_id: str
    event_type: EventType
    customer_id: str
    product_id: str
    timestamp: datetime
    amount: Optional[float] = None
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None

class RealtimeEventsResponse(BaseResponse):
    events: List[Event]

# ML models
class MLPrediction(BaseModel):
    predicted_sales: Optional[float] = None
    predicted_segment: Optional[str] = None
    churn_probability: Optional[float] = None
    confidence: float
    model_version: str

class MLPredictionRequest(BaseModel):
    model_name: str
    input_data: Dict[str, Any]
    prediction_type: str = "inference"

class MLPredictionResponse(BaseResponse):
    model_name: str
    prediction: MLPrediction

class ModelPerformance(BaseModel):
    model_name: str
    accuracy: float
    latency_ms: float
    total_predictions: int
    status: ModelStatus

class ModelStatusResponse(BaseResponse):
    models: List[ModelPerformance]

# Pipeline models
class Pipeline(BaseModel):
    name: str
    status: PipelineStatus
    last_run: datetime
    next_run: datetime
    success_rate: float

class PipelineStatusResponse(BaseResponse):
    pipelines: List[Pipeline]

class PipelineTriggerRequest(BaseModel):
    pipeline_name: str
    parameters: Optional[Dict[str, Any]] = None

class PipelineTriggerResponse(BaseResponse):
    message: str
    pipeline_name: str

# Monitoring models
class Alert(BaseModel):
    alert_type: str
    severity: SeverityLevel
    message: str
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime
    created_at: datetime

class AlertsResponse(BaseResponse):
    alerts: List[Alert]

class SystemMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    timestamp: datetime

class SystemMetricsResponse(BaseResponse):
    metrics: SystemMetrics

# Export models
class ExportRequest(BaseModel):
    data_type: str
    format: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    filters: Optional[Dict[str, Any]] = None

class ExportResponse(BaseResponse):
    export_id: str
    format: str
    status: str
    download_url: Optional[str] = None

# Authentication models
class User(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    permissions: List[str]

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseResponse):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User

class TokenRequest(BaseModel):
    access_token: str

class TokenResponse(BaseResponse):
    valid: bool
    user: Optional[User] = None

# Configuration models
class AlertConfig(BaseModel):
    alert_type: str
    threshold: float
    enabled: bool = True
    notification_channels: List[str] = ["email", "slack"]

class AlertConfigResponse(BaseResponse):
    config: AlertConfig

class SystemConfig(BaseModel):
    data_retention_days: int
    real_time_processing: bool
    ml_model_auto_retrain: bool
    alert_notifications: bool

class SystemConfigResponse(BaseResponse):
    config: SystemConfig

# Dashboard models
class DashboardWidget(BaseModel):
    widget_id: str
    widget_type: str
    title: str
    data: Dict[str, Any]
    position: Dict[str, int]
    size: Dict[str, int]

class Dashboard(BaseModel):
    dashboard_id: str
    name: str
    description: Optional[str] = None
    widgets: List[DashboardWidget]
    created_at: datetime
    updated_at: datetime

class DashboardResponse(BaseResponse):
    dashboard: Dashboard

class DashboardListResponse(BaseResponse):
    dashboards: List[Dashboard]

# Data quality models
class DataQualityMetric(BaseModel):
    metric_name: str
    value: float
    threshold: float
    status: str  # "pass", "warning", "fail"
    description: str

class DataQualityReport(BaseModel):
    report_id: str
    generated_at: datetime
    overall_score: float
    metrics: List[DataQualityMetric]
    recommendations: List[str]

class DataQualityResponse(BaseResponse):
    report: DataQualityReport

# Customer analytics models
class CustomerProfile(BaseModel):
    customer_id: str
    segment: CustomerSegment
    total_spent: float
    total_orders: int
    avg_order_value: float
    last_order_date: Optional[date] = None
    days_since_last_order: Optional[int] = None
    lifetime_value: float
    churn_risk: str  # "low", "medium", "high"

class CustomerProfileResponse(BaseResponse):
    profile: CustomerProfile

class CustomerListResponse(BaseResponse):
    customers: List[CustomerProfile]
    total_count: int
    page: int
    page_size: int

# Product analytics models
class ProductAnalytics(BaseModel):
    product_id: str
    product_name: str
    category: str
    total_sales: float
    total_quantity: int
    total_orders: int
    profit_margin: float
    last_sale_date: Optional[date] = None
    trend: str  # "increasing", "decreasing", "stable"

class ProductAnalyticsResponse(BaseResponse):
    analytics: ProductAnalytics

class ProductListResponse(BaseResponse):
    products: List[ProductAnalytics]
    total_count: int
    page: int
    page_size: int

# Geographic analytics models
class GeographicMetrics(BaseModel):
    region: str
    total_revenue: float
    total_orders: int
    unique_customers: int
    avg_order_value: float

class GeographicAnalyticsResponse(BaseResponse):
    metrics: List[GeographicMetrics]

# Time series models
class TimeSeriesDataPoint(BaseModel):
    timestamp: datetime
    value: float
    label: Optional[str] = None

class TimeSeriesData(BaseModel):
    series_name: str
    data_points: List[TimeSeriesDataPoint]
    metadata: Optional[Dict[str, Any]] = None

class TimeSeriesResponse(BaseResponse):
    time_series: List[TimeSeriesData]

# Validation models
class ValidationRule(BaseModel):
    rule_name: str
    field_name: str
    rule_type: str  # "range", "format", "required", "custom"
    parameters: Dict[str, Any]
    error_message: str

class ValidationResult(BaseModel):
    field_name: str
    is_valid: bool
    error_message: Optional[str] = None

class DataValidationResponse(BaseResponse):
    validation_results: List[ValidationResult]
    overall_valid: bool

# Custom validators
class CustomValidators:
    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
        """Validate that start_date is before end_date"""
        return start_date < end_date
    
    @staticmethod
    def validate_positive_number(value: float) -> bool:
        """Validate that value is positive"""
        return value > 0
    
    @staticmethod
    def validate_percentage(value: float) -> bool:
        """Validate that value is between 0 and 100"""
        return 0 <= value <= 100

# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    sales_summary = SalesSummary(
        total_orders=1000,
        total_revenue=50000.0,
        avg_order_value=50.0,
        unique_customers=500,
        total_profit=10000.0,
        avg_margin=20.0,
        period={"start_date": "2024-01-01", "end_date": "2024-01-31"}
    )
    
    print("Sales Summary created successfully:")
    print(sales_summary.json(indent=2))
