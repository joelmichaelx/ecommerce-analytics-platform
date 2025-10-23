#!/bin/bash

# Deploy E-commerce Analytics Platform to Vercel

echo "🚀 Deploying E-commerce Sales Analytics Platform to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI is not installed. Installing..."
    npm install -g vercel
fi

# Check if user is logged in to Vercel
if ! vercel whoami &> /dev/null; then
    echo "🔐 Please login to Vercel:"
    vercel login
fi

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git add .
    git commit -m "E-commerce Analytics Platform for Vercel deployment"
fi

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
vercel --prod

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📊 Your E-commerce Analytics Platform is now live!"
echo "🌐 Dashboard: https://your-app-name.vercel.app"
echo "🔧 API: https://your-app-name.vercel.app/api/analytics/sales/summary"
echo "📖 API Docs: https://your-app-name.vercel.app/api/health"
echo ""
echo "🔗 Check your Vercel dashboard for the exact URLs"
