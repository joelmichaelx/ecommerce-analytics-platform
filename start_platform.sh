#!/bin/bash

# E-commerce Sales Analytics Platform Startup Script

echo "🚀 Starting E-commerce Sales Analytics Platform..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, fastapi, pandas, numpy, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing required packages..."
    pip3 install streamlit fastapi uvicorn pandas numpy plotly
fi

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f streamlit 2>/dev/null
pkill -f "python.*simple_api" 2>/dev/null

# Start the API server
echo "🔧 Starting API server on port 8000..."
python3 simple_api.py &
API_PID=$!

# Wait a moment for API to start
sleep 2

# Start the Streamlit dashboard
echo "📊 Starting Streamlit dashboard on port 8501..."
streamlit run simple_dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true &
DASHBOARD_PID=$!

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 5

# Test the services
echo "🧪 Testing services..."

# Test API
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API server is running at http://localhost:8000"
    echo "   📖 API Documentation: http://localhost:8000/docs"
else
    echo "❌ API server failed to start"
fi

# Test Dashboard
if curl -s -I http://localhost:8501 > /dev/null; then
    echo "✅ Streamlit dashboard is running at http://localhost:8501"
else
    echo "❌ Streamlit dashboard failed to start"
fi

echo ""
echo "🎉 E-commerce Sales Analytics Platform is ready!"
echo ""
echo "📊 Dashboard: http://localhost:8501"
echo "🔧 API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep the script running and handle Ctrl+C
trap 'echo ""; echo "🛑 Stopping services..."; kill $API_PID $DASHBOARD_PID 2>/dev/null; exit 0' INT

# Wait for user to stop
wait
