"""
Simple E-commerce Analytics API
A working FastAPI application for E-commerce Sales Analytics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import random

# Create FastAPI app
app = FastAPI(
    title="E-commerce Sales Analytics API",
    description="Simple API for E-commerce Sales Analytics Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Generate sample data
def generate_sample_sales_data():
    """Generate sample sales data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    sales_data = []
    for i, date in enumerate(dates):
        base_sales = 1000 + i * 2
        seasonality = 200 * np.sin(2 * np.pi * i / 365)
        weekly_pattern = 100 * np.sin(2 * np.pi * i / 7)
        noise = np.random.normal(0, 100)
        
        daily_revenue = base_sales + seasonality + weekly_pattern + noise
        daily_orders = int(daily_revenue / np.random.uniform(50, 150))
        daily_customers = int(daily_orders * np.random.uniform(0.6, 0.9))
        
        sales_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'revenue': max(0, daily_revenue),
            'orders': max(1, daily_orders),
            'customers': max(1, daily_customers),
            'avg_order_value': daily_revenue / max(1, daily_orders)
        })
    
    return sales_data

def generate_sample_products():
    """Generate sample product data"""
    products = [
        'iPhone 15 Pro', 'Samsung Galaxy S24', 'MacBook Pro', 'Dell XPS 13',
        'AirPods Pro', 'Sony WH-1000XM5', 'iPad Air', 'Surface Pro 9',
        'Nintendo Switch', 'PlayStation 5', 'Xbox Series X', 'Apple Watch'
    ]
    
    categories = ['Electronics', 'Computers', 'Audio', 'Gaming', 'Wearables']
    
    product_data = []
    for product in products:
        category = random.choice(categories)
        revenue = np.random.uniform(10000, 500000)
        orders = int(revenue / np.random.uniform(100, 1000))
        margin = np.random.uniform(0.1, 0.4)
        
        product_data.append({
            'product': product,
            'category': category,
            'revenue': revenue,
            'orders': orders,
            'margin': margin,
            'profit': revenue * margin
        })
    
    return product_data

# Store sample data
sales_data = generate_sample_sales_data()
products_data = generate_sample_products()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "E-commerce Sales Analytics API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/analytics/sales/summary")
async def get_sales_summary(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31"
):
    """Get sales summary analytics"""
    try:
        # Filter data by date range
        filtered_data = [
            item for item in sales_data 
            if start_date <= item['date'] <= end_date
        ]
        
        if not filtered_data:
            return {"error": "No data found for the specified date range"}
        
        total_revenue = sum(item['revenue'] for item in filtered_data)
        total_orders = sum(item['orders'] for item in filtered_data)
        total_customers = sum(item['customers'] for item in filtered_data)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        return {
            "total_revenue": round(total_revenue, 2),
            "total_orders": total_orders,
            "total_customers": total_customers,
            "avg_order_value": round(avg_order_value, 2),
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "data_points": len(filtered_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/sales/trends")
async def get_sales_trends(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    granularity: str = "daily"
):
    """Get sales trends over time"""
    try:
        # Filter data by date range
        filtered_data = [
            item for item in sales_data 
            if start_date <= item['date'] <= end_date
        ]
        
        if not filtered_data:
            return {"error": "No data found for the specified date range"}
        
        # Group by granularity (simplified - just return daily for now)
        trends = []
        for item in filtered_data:
            trends.append({
                "date": item['date'],
                "revenue": item['revenue'],
                "orders": item['orders'],
                "customers": item['customers']
            })
        
        return {"trends": trends}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/products/top")
async def get_top_products(
    limit: int = 10,
    metric: str = "revenue"
):
    """Get top performing products"""
    try:
        # Sort products by metric
        if metric == "revenue":
            sorted_products = sorted(products_data, key=lambda x: x['revenue'], reverse=True)
        elif metric == "orders":
            sorted_products = sorted(products_data, key=lambda x: x['orders'], reverse=True)
        elif metric == "profit":
            sorted_products = sorted(products_data, key=lambda x: x['profit'], reverse=True)
        else:
            sorted_products = sorted(products_data, key=lambda x: x['revenue'], reverse=True)
        
        # Return top N products
        top_products = sorted_products[:limit]
        
        return {"top_products": top_products}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/customers/segments")
async def get_customer_segments():
    """Get customer segmentation analysis"""
    try:
        segments = [
            {
                "segment": "Champions",
                "count": 150,
                "avg_spent": 2500,
                "avg_orders": 25,
                "total_value": 375000
            },
            {
                "segment": "Loyal Customers",
                "count": 300,
                "avg_spent": 1200,
                "avg_orders": 15,
                "total_value": 360000
            },
            {
                "segment": "New Customers",
                "count": 200,
                "avg_spent": 300,
                "avg_orders": 3,
                "total_value": 60000
            },
            {
                "segment": "At Risk",
                "count": 100,
                "avg_spent": 800,
                "avg_orders": 8,
                "total_value": 80000
            },
            {
                "segment": "Cannot Lose Them",
                "count": 50,
                "avg_spent": 5000,
                "avg_orders": 50,
                "total_value": 250000
            }
        ]
        
        return {"customer_segments": segments}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/realtime/metrics")
async def get_realtime_metrics():
    """Get real-time metrics"""
    try:
        metrics = {
            "total_events": random.randint(1000, 10000),
            "total_revenue": round(random.uniform(50000, 200000), 2),
            "conversion_rate": round(random.uniform(2, 8), 2),
            "avg_order_value": round(random.uniform(80, 200), 2),
            "events_by_type": {
                "purchase": random.randint(100, 1000),
                "view": random.randint(1000, 5000),
                "add_to_cart": random.randint(200, 1500),
                "remove_from_cart": random.randint(50, 300)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/predict/sales")
async def predict_sales(
    days_ahead: int = 7
):
    """Predict sales for the next N days"""
    try:
        # Simple prediction based on recent trends
        recent_sales = sales_data[-7:]  # Last 7 days
        avg_daily_sales = sum(item['revenue'] for item in recent_sales) / len(recent_sales)
        
        predictions = []
        for i in range(1, days_ahead + 1):
            # Add some trend and seasonality
            trend = avg_daily_sales * (1 + i * 0.01)  # 1% growth per day
            seasonality = avg_daily_sales * 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
            noise = avg_daily_sales * 0.05 * np.random.normal(0, 1)  # 5% noise
            
            predicted_sales = trend + seasonality + noise
            
            predictions.append({
                "date": (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                "predicted_revenue": round(max(0, predicted_sales), 2),
                "confidence": round(random.uniform(0.7, 0.95), 2)
            })
        
        return {
            "predictions": predictions,
            "model_version": "1.0.0",
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/predict/churn")
async def predict_churn(
    customer_id: str = "CUST_001"
):
    """Predict customer churn probability"""
    try:
        # Simple churn prediction
        churn_probability = random.uniform(0.1, 0.8)
        
        if churn_probability < 0.3:
            risk_level = "Low"
        elif churn_probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "customer_id": customer_id,
            "churn_probability": round(churn_probability, 3),
            "risk_level": risk_level,
            "confidence": round(random.uniform(0.8, 0.95), 2),
            "recommendations": [
                "Send personalized email campaign",
                "Offer discount on next purchase",
                "Provide excellent customer service"
            ],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
