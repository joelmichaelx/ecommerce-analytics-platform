"""
FastAPI Main Application
Comprehensive API for E-commerce Sales Analytics Platform
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import redis
import json
from pydantic import BaseModel, Field
import asyncio
from contextlib import asynccontextmanager

# Import custom modules
from .models import *
from .auth import verify_token
from .database import get_db_connection
from .cache import get_redis_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic models for request/response
class SalesQuery(BaseModel):
    start_date: datetime
    end_date: datetime
    category: Optional[str] = None
    customer_segment: Optional[str] = None
    limit: Optional[int] = 100

class PredictionRequest(BaseModel):
    customer_id: str
    product_id: Optional[str] = None
    features: Dict[str, Any] = {}

class AlertConfig(BaseModel):
    alert_type: str
    threshold: float
    enabled: bool = True

class MLModelRequest(BaseModel):
    model_name: str
    input_data: Dict[str, Any]
    prediction_type: str = "inference"

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting E-commerce Analytics API")
    yield
    # Shutdown
    logger.info("Shutting down E-commerce Analytics API")

# Create FastAPI app
app = FastAPI(
    title="E-commerce Sales Analytics API",
    description="Comprehensive API for E-commerce Sales Analytics Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user = await verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

# Analytics endpoints
@app.get("/analytics/sales/summary")
async def get_sales_summary(
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    category: Optional[str] = Query(None, description="Product category filter"),
    current_user: dict = Depends(get_current_user)
):
    """Get sales summary analytics"""
    try:
        db = get_db_connection()
        
        # Build query based on filters
        query = """
        SELECT 
            COUNT(DISTINCT fs.order_id) as total_orders,
            SUM(fs.total_amount) as total_revenue,
            AVG(fs.total_amount) as avg_order_value,
            COUNT(DISTINCT fs.customer_key) as unique_customers,
            SUM(fs.profit_amount) as total_profit,
            AVG(fs.margin_percentage) as avg_margin
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        JOIN processed_data.dim_products dp ON fs.product_key = dp.product_key
        WHERE dd.full_date BETWEEN :start_date AND :end_date
        """
        
        params = {
            'start_date': start_date.date(),
            'end_date': end_date.date()
        }
        
        if category:
            query += " AND dp.category = :category"
            params['category'] = category
        
        result = db.execute(text(query), params).fetchone()
        
        return {
            "total_orders": int(result.total_orders or 0),
            "total_revenue": float(result.total_revenue or 0),
            "avg_order_value": float(result.avg_order_value or 0),
            "unique_customers": int(result.unique_customers or 0),
            "total_profit": float(result.total_profit or 0),
            "avg_margin": float(result.avg_margin or 0),
            "period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get sales summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/sales/trends")
async def get_sales_trends(
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    granularity: str = Query("daily", description="Time granularity (daily, weekly, monthly)"),
    current_user: dict = Depends(get_current_user)
):
    """Get sales trends over time"""
    try:
        db = get_db_connection()
        
        # Determine date grouping based on granularity
        if granularity == "daily":
            date_group = "dd.full_date"
        elif granularity == "weekly":
            date_group = "DATE_TRUNC('week', dd.full_date)"
        elif granularity == "monthly":
            date_group = "DATE_TRUNC('month', dd.full_date)"
        else:
            raise HTTPException(status_code=400, detail="Invalid granularity")
        
        query = f"""
        SELECT 
            {date_group} as period,
            COUNT(DISTINCT fs.order_id) as orders,
            SUM(fs.total_amount) as revenue,
            COUNT(DISTINCT fs.customer_key) as customers
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date BETWEEN :start_date AND :end_date
        GROUP BY {date_group}
        ORDER BY period
        """
        
        result = db.execute(text(query), {
            'start_date': start_date.date(),
            'end_date': end_date.date()
        }).fetchall()
        
        trends = []
        for row in result:
            trends.append({
                "period": str(row.period),
                "orders": int(row.orders),
                "revenue": float(row.revenue),
                "customers": int(row.customers)
            })
        
        return {"trends": trends}
        
    except Exception as e:
        logger.error(f"Failed to get sales trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/products/top")
async def get_top_products(
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    limit: int = Query(10, description="Number of top products to return"),
    metric: str = Query("revenue", description="Metric to rank by (revenue, quantity, orders)"),
    current_user: dict = Depends(get_current_user)
):
    """Get top performing products"""
    try:
        db = get_db_connection()
        
        # Determine metric column
        if metric == "revenue":
            metric_col = "SUM(fs.total_amount)"
        elif metric == "quantity":
            metric_col = "SUM(fs.quantity)"
        elif metric == "orders":
            metric_col = "COUNT(DISTINCT fs.order_id)"
        else:
            raise HTTPException(status_code=400, detail="Invalid metric")
        
        query = f"""
        SELECT 
            dp.product_name,
            dp.category,
            dp.brand,
            {metric_col} as metric_value,
            SUM(fs.total_amount) as total_revenue,
            SUM(fs.quantity) as total_quantity,
            COUNT(DISTINCT fs.order_id) as total_orders
        FROM processed_data.fact_sales fs
        JOIN processed_data.dim_products dp ON fs.product_key = dp.product_key
        JOIN processed_data.dim_date dd ON fs.date_key = dd.date_key
        WHERE dd.full_date BETWEEN :start_date AND :end_date
        AND dp.current_flag = TRUE
        GROUP BY dp.product_id, dp.product_name, dp.category, dp.brand
        ORDER BY metric_value DESC
        LIMIT :limit
        """
        
        result = db.execute(text(query), {
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'limit': limit
        }).fetchall()
        
        products = []
        for row in result:
            products.append({
                "product_name": row.product_name,
                "category": row.category,
                "brand": row.brand,
                "metric_value": float(row.metric_value),
                "total_revenue": float(row.total_revenue),
                "total_quantity": int(row.total_quantity),
                "total_orders": int(row.total_orders)
            })
        
        return {"top_products": products}
        
    except Exception as e:
        logger.error(f"Failed to get top products: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/customers/segments")
async def get_customer_segments(
    current_user: dict = Depends(get_current_user)
):
    """Get customer segmentation analysis"""
    try:
        db = get_db_connection()
        
        query = """
        SELECT 
            customer_segment,
            COUNT(*) as customer_count,
            AVG(total_spent) as avg_spent,
            AVG(total_orders) as avg_orders,
            AVG(avg_order_value) as avg_order_value
        FROM analytics.customer_segments
        GROUP BY customer_segment
        ORDER BY customer_count DESC
        """
        
        result = db.execute(text(query)).fetchall()
        
        segments = []
        for row in result:
            segments.append({
                "segment": row.customer_segment,
                "customer_count": int(row.customer_count),
                "avg_spent": float(row.avg_spent),
                "avg_orders": float(row.avg_orders),
                "avg_order_value": float(row.avg_order_value)
            })
        
        return {"customer_segments": segments}
        
    except Exception as e:
        logger.error(f"Failed to get customer segments: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Real-time analytics endpoints
@app.get("/realtime/metrics")
async def get_realtime_metrics(
    current_user: dict = Depends(get_current_user)
):
    """Get real-time metrics from Redis"""
    try:
        redis_client = get_redis_connection()
        
        metrics = {
            "total_events": int(redis_client.get("total_events") or 0),
            "total_revenue": float(redis_client.get("total_revenue") or 0),
            "conversion_rate": 0,
            "avg_order_value": 0,
            "events_by_type": {}
        }
        
        # Get event type breakdown
        event_types = ['purchase', 'view', 'add_to_cart', 'remove_from_cart']
        for event_type in event_types:
            count = int(redis_client.get(f"events:{event_type}") or 0)
            metrics["events_by_type"][event_type] = count
        
        # Calculate conversion rate
        total_views = metrics["events_by_type"].get("view", 0)
        total_purchases = metrics["events_by_type"].get("purchase", 0)
        if total_views > 0:
            metrics["conversion_rate"] = (total_purchases / total_views) * 100
        
        # Calculate average order value
        if total_purchases > 0:
            metrics["avg_order_value"] = metrics["total_revenue"] / total_purchases
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get real-time metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/realtime/events")
async def get_realtime_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(100, description="Number of events to return"),
    current_user: dict = Depends(get_current_user)
):
    """Get recent real-time events"""
    try:
        redis_client = get_redis_connection()
        
        # Get recent events from Redis (this would be implemented based on your Redis structure)
        events = []
        
        # This is a placeholder - in practice, you'd retrieve events from Redis streams or lists
        for i in range(min(limit, 10)):  # Placeholder data
            events.append({
                "event_id": f"EVENT_{i:08d}",
                "event_type": "purchase",
                "customer_id": f"CUST_{i:06d}",
                "product_id": f"PROD_{i:06d}",
                "timestamp": datetime.now().isoformat(),
                "amount": 100.0 + i * 10
            })
        
        return {"events": events}
        
    except Exception as e:
        logger.error(f"Failed to get real-time events: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ML model endpoints
@app.post("/ml/predict")
async def predict(
    request: MLModelRequest,
    current_user: dict = Depends(get_current_user)
):
    """Make ML model predictions"""
    try:
        # This would integrate with your ML models
        # For now, returning placeholder predictions
        
        if request.model_name == "sales_forecasting":
            prediction = {
                "predicted_sales": 1000.0 + np.random.normal(0, 100),
                "confidence": 0.85,
                "model_version": "1.0.0"
            }
        elif request.model_name == "customer_segmentation":
            prediction = {
                "predicted_segment": "Loyal Customer",
                "confidence": 0.92,
                "model_version": "1.0.0"
            }
        elif request.model_name == "churn_prediction":
            prediction = {
                "churn_probability": 0.15,
                "risk_level": "Low",
                "confidence": 0.88,
                "model_version": "1.0.0"
            }
        else:
            raise HTTPException(status_code=400, detail="Unknown model")
        
        return {
            "model_name": request.model_name,
            "prediction": prediction,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to make prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/ml/models/status")
async def get_model_status(
    current_user: dict = Depends(get_current_user)
):
    """Get ML model status and performance"""
    try:
        redis_client = get_redis_connection()
        
        models = ["sales_forecasting", "customer_segmentation", "churn_prediction"]
        model_status = []
        
        for model in models:
            accuracy = float(redis_client.get(f"ml_accuracy:{model}") or 0)
            latency = float(redis_client.get(f"ml_latency:{model}") or 0)
            predictions = int(redis_client.get(f"ml_predictions:{model}") or 0)
            
            model_status.append({
                "model_name": model,
                "accuracy": accuracy,
                "latency_ms": latency * 1000,
                "total_predictions": predictions,
                "status": "active" if accuracy > 0.8 else "degraded"
            })
        
        return {"models": model_status}
        
    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Data pipeline endpoints
@app.get("/pipeline/status")
async def get_pipeline_status(
    current_user: dict = Depends(get_current_user)
):
    """Get data pipeline status"""
    try:
        # This would check Airflow or other pipeline systems
        pipelines = [
            {
                "name": "data_ingestion",
                "status": "running",
                "last_run": datetime.now() - timedelta(hours=1),
                "next_run": datetime.now() + timedelta(hours=23),
                "success_rate": 0.95
            },
            {
                "name": "data_processing",
                "status": "completed",
                "last_run": datetime.now() - timedelta(hours=2),
                "next_run": datetime.now() + timedelta(hours=22),
                "success_rate": 0.98
            },
            {
                "name": "ml_training",
                "status": "scheduled",
                "last_run": datetime.now() - timedelta(days=1),
                "next_run": datetime.now() + timedelta(hours=6),
                "success_rate": 0.90
            }
        ]
        
        return {"pipelines": pipelines}
        
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/pipeline/trigger")
async def trigger_pipeline(
    pipeline_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Trigger a data pipeline"""
    try:
        # This would trigger the actual pipeline
        # For now, returning a success response
        
        return {
            "message": f"Pipeline {pipeline_name} triggered successfully",
            "pipeline_name": pipeline_name,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Monitoring endpoints
@app.get("/monitoring/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, description="Number of alerts to return"),
    current_user: dict = Depends(get_current_user)
):
    """Get system alerts"""
    try:
        db = get_db_connection()
        
        query = """
        SELECT 
            alert_type,
            severity,
            message,
            current_value,
            previous_value,
            threshold,
            timestamp,
            created_at
        FROM alerts
        WHERE 1=1
        """
        
        params = {}
        if severity:
            query += " AND severity = :severity"
            params['severity'] = severity
        
        query += " ORDER BY created_at DESC LIMIT :limit"
        params['limit'] = limit
        
        result = db.execute(text(query), params).fetchall()
        
        alerts = []
        for row in result:
            alerts.append({
                "alert_type": row.alert_type,
                "severity": row.severity,
                "message": row.message,
                "current_value": row.current_value,
                "previous_value": row.previous_value,
                "threshold": row.threshold,
                "timestamp": row.timestamp,
                "created_at": row.created_at
            })
        
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/monitoring/metrics")
async def get_system_metrics(
    current_user: dict = Depends(get_current_user)
):
    """Get system performance metrics"""
    try:
        import psutil
        
        metrics = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": datetime.now()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Export endpoints
@app.get("/export/data")
async def export_data(
    data_type: str = Query(..., description="Type of data to export"),
    format: str = Query("json", description="Export format (json, csv)"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    current_user: dict = Depends(get_current_user)
):
    """Export data in various formats"""
    try:
        db = get_db_connection()
        
        # This would implement actual data export
        # For now, returning a placeholder response
        
        return {
            "message": f"Data export initiated for {data_type}",
            "export_id": f"EXPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "format": format,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to export data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
