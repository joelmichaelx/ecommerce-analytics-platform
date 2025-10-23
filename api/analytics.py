"""
Vercel API Routes for E-commerce Analytics
Serverless functions for analytics endpoints
"""

from http.server import BaseHTTPRequestHandler
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        try:
            # Parse URL path
            path = self.path.split('?')[0]
            
            if path == '/api/analytics/sales/summary':
                response = self.get_sales_summary()
            elif path == '/api/analytics/sales/trends':
                response = self.get_sales_trends()
            elif path == '/api/analytics/products/top':
                response = self.get_top_products()
            elif path == '/api/analytics/customers/segments':
                response = self.get_customer_segments()
            elif path == '/api/realtime/metrics':
                response = self.get_realtime_metrics()
            elif path == '/api/health':
                response = {"status": "healthy", "timestamp": datetime.now().isoformat()}
            else:
                response = {"error": "Endpoint not found"}
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return
            
            # Send successful response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {"error": str(e)}
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def get_sales_summary(self):
        """Get sales summary analytics"""
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        
        total_revenue = 0
        total_orders = 0
        total_customers = 0
        
        for i, date in enumerate(dates):
            base_sales = 1000 + i * 2
            seasonality = 200 * np.sin(2 * np.pi * i / 365)
            weekly_pattern = 100 * np.sin(2 * np.pi * i / 7)
            noise = np.random.normal(0, 100)
            
            daily_revenue = max(0, base_sales + seasonality + weekly_pattern + noise)
            daily_orders = int(daily_revenue / np.random.uniform(50, 150))
            daily_customers = int(daily_orders * np.random.uniform(0.6, 0.9))
            
            total_revenue += daily_revenue
            total_orders += daily_orders
            total_customers += daily_customers
        
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        return {
            "total_revenue": round(total_revenue, 2),
            "total_orders": total_orders,
            "total_customers": total_customers,
            "avg_order_value": round(avg_order_value, 2),
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }
    
    def get_sales_trends(self):
        """Get sales trends over time"""
        # Generate sample trends data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        
        trends = []
        for i, date in enumerate(dates[:30]):  # Last 30 days
            base_sales = 1000 + i * 2
            seasonality = 200 * np.sin(2 * np.pi * i / 365)
            weekly_pattern = 100 * np.sin(2 * np.pi * i / 7)
            noise = np.random.normal(0, 100)
            
            daily_revenue = max(0, base_sales + seasonality + weekly_pattern + noise)
            daily_orders = int(daily_revenue / np.random.uniform(50, 150))
            daily_customers = int(daily_orders * np.random.uniform(0.6, 0.9))
            
            trends.append({
                "date": date.strftime('%Y-%m-%d'),
                "revenue": round(daily_revenue, 2),
                "orders": daily_orders,
                "customers": daily_customers
            })
        
        return {"trends": trends}
    
    def get_top_products(self):
        """Get top performing products"""
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
                "product": product,
                "category": category,
                "revenue": round(revenue, 2),
                "orders": orders,
                "margin": round(margin, 2),
                "profit": round(revenue * margin, 2)
            })
        
        # Sort by revenue and return top 10
        product_data.sort(key=lambda x: x['revenue'], reverse=True)
        return {"top_products": product_data[:10]}
    
    def get_customer_segments(self):
        """Get customer segmentation analysis"""
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
    
    def get_realtime_metrics(self):
        """Get real-time metrics"""
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
