"""
Vercel API endpoints for E-commerce Analytics Platform
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

def generate_sample_data():
    """Generate comprehensive sample data for the analytics platform"""
    np.random.seed(42)
    
    # Generate sales data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    sales_data = []
    
    for i, date in enumerate(dates):
        base_sales = 1000 + i * 2
        seasonality = 200 * np.sin(2 * np.pi * i / 365)
        weekly_pattern = 100 * np.sin(2 * np.pi * i / 7)
        noise = np.random.normal(0, 100)
        
        daily_revenue = max(0, base_sales + seasonality + weekly_pattern + noise)
        daily_orders = int(daily_revenue / np.random.uniform(50, 150))
        daily_customers = int(daily_orders * np.random.uniform(0.6, 0.9))
        
        sales_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'revenue': daily_revenue,
            'orders': daily_orders,
            'customers': daily_customers,
            'avg_order_value': daily_revenue / max(1, daily_orders)
        })
    
    return sales_data

def generate_product_data():
    """Generate product performance data"""
    products = [
        'iPhone 15 Pro', 'Samsung Galaxy S24', 'MacBook Pro', 'Dell XPS 13',
        'AirPods Pro', 'Sony WH-1000XM5', 'iPad Air', 'Surface Pro 9',
        'Nintendo Switch', 'PlayStation 5', 'Xbox Series X', 'Apple Watch',
        'Fitbit Versa', 'GoPro Hero', 'Canon EOS R5', 'Nikon Z9',
        'Tesla Model 3', 'BMW i3', 'Mercedes EQS', 'Audi e-tron'
    ]
    
    categories = ['Electronics', 'Computers', 'Audio', 'Gaming', 'Wearables', 'Cameras', 'Automotive']
    
    product_data = []
    for product in products:
        category = np.random.choice(categories)
        revenue = np.random.uniform(10000, 500000)
        orders = int(revenue / np.random.uniform(100, 1000))
        margin = np.random.uniform(0.1, 0.5)
        
        product_data.append({
            'product': product,
            'category': category,
            'revenue': revenue,
            'orders': orders,
            'margin': margin,
            'profit': revenue * margin
        })
    
    return product_data

def generate_customer_segments():
    """Generate customer segmentation data"""
    return [
        {'segment': 'Champions', 'count': 150, 'avg_spent': 2500, 'avg_orders': 25, 'total_value': 375000},
        {'segment': 'Loyal Customers', 'count': 300, 'avg_spent': 1200, 'avg_orders': 15, 'total_value': 360000},
        {'segment': 'New Customers', 'count': 200, 'avg_spent': 300, 'avg_orders': 3, 'total_value': 60000},
        {'segment': 'At Risk', 'count': 100, 'avg_spent': 800, 'avg_orders': 8, 'total_value': 80000},
        {'segment': 'Cannot Lose Them', 'count': 50, 'avg_spent': 5000, 'avg_orders': 50, 'total_value': 250000}
    ]

def handler(request):
    """Main handler for Vercel API requests"""
    path = request.get('path', '')
    
    if path == '/api/analytics/sales/summary':
        sales_data = generate_sample_data()
        total_revenue = sum(d['revenue'] for d in sales_data)
        total_orders = sum(d['orders'] for d in sales_data)
        total_customers = sum(d['customers'] for d in sales_data)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'total_revenue': total_revenue,
                'total_orders': total_orders,
                'total_customers': total_customers,
                'avg_order_value': avg_order_value
            })
        }
    
    elif path == '/api/analytics/sales/trends':
        sales_data = generate_sample_data()
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'trends': sales_data})
        }
    
    elif path == '/api/analytics/products/top':
        product_data = generate_product_data()
        top_products = sorted(product_data, key=lambda x: x['revenue'], reverse=True)[:10]
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'top_products': top_products})
        }
    
    elif path == '/api/analytics/customers/segments':
        segments = generate_customer_segments()
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'customer_segments': segments})
        }
    
    elif path == '/api/realtime/metrics':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'total_events': np.random.randint(100, 1000),
                'total_revenue': np.random.uniform(1000, 5000),
                'conversion_rate': np.random.uniform(2, 8),
                'avg_order_value': np.random.uniform(80, 200)
            })
        }
    
    elif path == '/api/health':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
        }
    
    else:
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Endpoint not found'})
        }
