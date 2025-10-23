"""
Customer Analytics and Segmentation
Advanced customer behavior analysis and segmentation models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CustomerAnalytics:
    """Advanced customer analytics and segmentation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.cluster_models = {}
        self.segment_profiles = {}
        
    def calculate_rfm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) metrics"""
        try:
            self.logger.info("Calculating RFM metrics")
            
            # Ensure date column is datetime
            df['order_date'] = pd.to_datetime(df['order_date'])
            
            # Calculate RFM metrics
            rfm_df = df.groupby('customer_id').agg({
                'order_date': lambda x: (df['order_date'].max() - x.max()).days,  # Recency
                'order_id': 'nunique',  # Frequency
                'total_amount': 'sum'   # Monetary
            }).reset_index()
            
            rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']
            
            # Calculate RFM scores (1-5 scale)
            rfm_df['r_score'] = pd.qcut(rfm_df['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
            rfm_df['f_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
            rfm_df['m_score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
            
            # Convert to numeric
            rfm_df['r_score'] = rfm_df['r_score'].astype(int)
            rfm_df['f_score'] = rfm_df['f_score'].astype(int)
            rfm_df['m_score'] = rfm_df['m_score'].astype(int)
            
            # Create RFM segment
            rfm_df['rfm_segment'] = rfm_df['r_score'].astype(str) + rfm_df['f_score'].astype(str) + rfm_df['m_score'].astype(str)
            
            # Define customer segments based on RFM
            def segment_customers(row):
                if row['r_score'] >= 4 and row['f_score'] >= 4 and row['m_score'] >= 4:
                    return 'Champions'
                elif row['r_score'] >= 3 and row['f_score'] >= 3 and row['m_score'] >= 3:
                    return 'Loyal Customers'
                elif row['r_score'] >= 4 and row['f_score'] <= 2:
                    return 'New Customers'
                elif row['r_score'] >= 3 and row['f_score'] >= 2 and row['m_score'] >= 3:
                    return 'Potential Loyalists'
                elif row['r_score'] >= 3 and row['f_score'] <= 2 and row['m_score'] <= 2:
                    return 'Promising'
                elif row['r_score'] <= 2 and row['f_score'] >= 3 and row['m_score'] >= 3:
                    return 'Need Attention'
                elif row['r_score'] <= 2 and row['f_score'] >= 2 and row['m_score'] >= 2:
                    return 'About to Sleep'
                elif row['r_score'] <= 2 and row['f_score'] <= 2 and row['m_score'] >= 3:
                    return 'At Risk'
                elif row['r_score'] <= 2 and row['f_score'] <= 2 and row['m_score'] <= 2:
                    return 'Cannot Lose Them'
                else:
                    return 'Others'
            
            rfm_df['customer_segment'] = rfm_df.apply(segment_customers, axis=1)
            
            self.logger.info(f"RFM analysis completed for {len(rfm_df)} customers")
            return rfm_df
            
        except Exception as e:
            self.logger.error(f"RFM calculation failed: {str(e)}")
            raise
    
    def calculate_customer_lifetime_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Customer Lifetime Value (CLV)"""
        try:
            self.logger.info("Calculating Customer Lifetime Value")
            
            # Calculate customer metrics
            customer_metrics = df.groupby('customer_id').agg({
                'order_date': ['min', 'max', 'nunique'],
                'total_amount': ['sum', 'mean', 'std'],
                'order_id': 'nunique',
                'product_id': 'nunique'
            }).reset_index()
            
            # Flatten column names
            customer_metrics.columns = ['customer_id', 'first_order_date', 'last_order_date', 
                                     'order_frequency', 'total_spent', 'avg_order_value', 
                                     'order_value_std', 'total_orders', 'unique_products']
            
            # Calculate customer age in days
            customer_metrics['customer_age_days'] = (
                customer_metrics['last_order_date'] - customer_metrics['first_order_date']
            ).dt.days
            
            # Calculate purchase frequency (orders per day)
            customer_metrics['purchase_frequency'] = customer_metrics['total_orders'] / (customer_metrics['customer_age_days'] + 1)
            
            # Calculate average days between orders
            customer_metrics['avg_days_between_orders'] = customer_metrics['customer_age_days'] / (customer_metrics['total_orders'] - 1)
            customer_metrics['avg_days_between_orders'] = customer_metrics['avg_days_between_orders'].fillna(0)
            
            # Calculate CLV using different methods
            
            # Method 1: Simple CLV (Total Revenue)
            customer_metrics['clv_simple'] = customer_metrics['total_spent']
            
            # Method 2: CLV with frequency and recency
            # CLV = (Average Order Value × Purchase Frequency) / Churn Rate
            # Assuming 10% monthly churn rate
            monthly_churn_rate = 0.1
            customer_metrics['clv_frequency'] = (
                customer_metrics['avg_order_value'] * 
                customer_metrics['purchase_frequency'] * 30 / monthly_churn_rate
            )
            
            # Method 3: CLV with customer age consideration
            # CLV = (Average Order Value × Purchase Frequency × Customer Age) / (1 + Discount Rate)
            discount_rate = 0.1
            customer_metrics['clv_age_adjusted'] = (
                customer_metrics['avg_order_value'] * 
                customer_metrics['purchase_frequency'] * 
                customer_metrics['customer_age_days'] / 365
            ) / (1 + discount_rate)
            
            # Method 4: Predictive CLV using RFM
            # This would typically use a machine learning model
            # For now, using a simplified formula
            customer_metrics['clv_predictive'] = (
                customer_metrics['total_spent'] * 
                (1 + customer_metrics['purchase_frequency']) * 
                (1 - customer_metrics['order_value_std'] / customer_metrics['avg_order_value'])
            )
            
            self.logger.info("Customer Lifetime Value calculation completed")
            return customer_metrics
            
        except Exception as e:
            self.logger.error(f"CLV calculation failed: {str(e)}")
            raise
    
    def customer_segmentation_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced customer segmentation using clustering algorithms"""
        try:
            self.logger.info("Starting customer segmentation clustering")
            
            # Prepare features for clustering
            features = ['recency', 'frequency', 'monetary', 'avg_order_value', 'purchase_frequency']
            
            # Handle missing values
            df_cluster = df[features].fillna(df[features].mean())
            
            # Scale features
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_cluster)
            self.scalers['customer_clustering'] = scaler
            
            # Try different clustering algorithms
            clustering_results = {}
            
            # K-Means clustering
            kmeans_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(df_scaled)
                silhouette_avg = silhouette_score(df_scaled, cluster_labels)
                kmeans_scores.append((k, silhouette_avg))
            
            # Find optimal number of clusters
            optimal_k = max(kmeans_scores, key=lambda x: x[1])[0]
            kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            kmeans_labels = kmeans_optimal.fit_predict(df_scaled)
            
            clustering_results['kmeans'] = {
                'model': kmeans_optimal,
                'labels': kmeans_labels,
                'n_clusters': optimal_k,
                'silhouette_score': silhouette_score(df_scaled, kmeans_labels),
                'calinski_harabasz_score': calinski_harabasz_score(df_scaled, kmeans_labels)
            }
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(df_scaled)
            
            if len(set(dbscan_labels)) > 1:  # Check if clusters were found
                clustering_results['dbscan'] = {
                    'model': dbscan,
                    'labels': dbscan_labels,
                    'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                    'silhouette_score': silhouette_score(df_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0
                }
            
            # Agglomerative clustering
            agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
            agg_labels = agg_clustering.fit_predict(df_scaled)
            
            clustering_results['agglomerative'] = {
                'model': agg_clustering,
                'labels': agg_labels,
                'n_clusters': optimal_k,
                'silhouette_score': silhouette_score(df_scaled, agg_labels),
                'calinski_harabasz_score': calinski_harabasz_score(df_scaled, agg_labels)
            }
            
            # Store the best clustering result
            best_method = max(clustering_results.keys(), 
                            key=lambda x: clustering_results[x]['silhouette_score'])
            self.cluster_models['customer_segmentation'] = clustering_results[best_method]
            
            # Create segment profiles
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = clustering_results[best_method]['labels']
            
            segment_profiles = df_with_clusters.groupby('cluster').agg({
                'recency': ['mean', 'std'],
                'frequency': ['mean', 'std'],
                'monetary': ['mean', 'std'],
                'avg_order_value': ['mean', 'std'],
                'purchase_frequency': ['mean', 'std'],
                'customer_id': 'count'
            }).round(2)
            
            # Flatten column names
            segment_profiles.columns = ['_'.join(col).strip() for col in segment_profiles.columns]
            segment_profiles = segment_profiles.reset_index()
            
            # Add segment names
            segment_names = {
                0: 'Low Value',
                1: 'Medium Value',
                2: 'High Value',
                3: 'Premium',
                4: 'Champions'
            }
            
            segment_profiles['segment_name'] = segment_profiles['cluster'].map(segment_names)
            self.segment_profiles = segment_profiles
            
            results = {
                'clustering_results': clustering_results,
                'best_method': best_method,
                'segment_profiles': segment_profiles,
                'df_with_clusters': df_with_clusters
            }
            
            self.logger.info(f"Customer segmentation completed using {best_method}")
            return results
            
        except Exception as e:
            self.logger.error(f"Customer segmentation failed: {str(e)}")
            raise
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous customer behavior"""
        try:
            self.logger.info("Detecting customer anomalies")
            
            # Prepare features for anomaly detection
            anomaly_features = ['recency', 'frequency', 'monetary', 'avg_order_value', 'purchase_frequency']
            df_anomaly = df[anomaly_features].fillna(df[anomaly_features].mean())
            
            # Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(df_anomaly)
            anomaly_scores = iso_forest.decision_function(df_anomaly)
            
            # Add anomaly information to dataframe
            df_anomaly_result = df.copy()
            df_anomaly_result['is_anomaly'] = anomaly_labels == -1
            df_anomaly_result['anomaly_score'] = anomaly_scores
            
            # Statistical anomaly detection
            for feature in anomaly_features:
                if feature in df_anomaly_result.columns:
                    # Z-score method
                    z_scores = np.abs(stats.zscore(df_anomaly_result[feature].fillna(0)))
                    df_anomaly_result[f'{feature}_zscore'] = z_scores
                    df_anomaly_result[f'{feature}_is_outlier'] = z_scores > 3
            
            # IQR method
            for feature in anomaly_features:
                if feature in df_anomaly_result.columns:
                    Q1 = df_anomaly_result[feature].quantile(0.25)
                    Q3 = df_anomaly_result[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_anomaly_result[f'{feature}_iqr_outlier'] = (
                        (df_anomaly_result[feature] < lower_bound) | 
                        (df_anomaly_result[feature] > upper_bound)
                    )
            
            self.logger.info(f"Anomaly detection completed. Found {df_anomaly_result['is_anomaly'].sum()} anomalies")
            return df_anomaly_result
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            raise
    
    def customer_churn_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict customer churn probability"""
        try:
            self.logger.info("Predicting customer churn")
            
            # Define churn (no orders in last 90 days)
            cutoff_date = df['order_date'].max() - timedelta(days=90)
            
            # Calculate churn indicators
            churn_features = df.groupby('customer_id').agg({
                'order_date': ['max', 'nunique'],
                'total_amount': ['sum', 'mean'],
                'order_id': 'nunique'
            }).reset_index()
            
            # Flatten column names
            churn_features.columns = ['customer_id', 'last_order_date', 'order_frequency', 
                                    'total_spent', 'avg_order_value', 'total_orders']
            
            # Calculate days since last order
            churn_features['days_since_last_order'] = (
                df['order_date'].max() - churn_features['last_order_date']
            ).dt.days
            
            # Define churn (binary)
            churn_features['is_churned'] = churn_features['days_since_last_order'] > 90
            
            # Calculate churn probability features
            churn_features['order_frequency_trend'] = churn_features['total_orders'] / (
                churn_features['days_since_last_order'] + 1
            )
            
            # Customer tenure
            customer_tenure = df.groupby('customer_id')['order_date'].agg(['min', 'max']).reset_index()
            customer_tenure['customer_tenure_days'] = (
                customer_tenure['max'] - customer_tenure['min']
            ).dt.days
            
            churn_features = churn_features.merge(
                customer_tenure[['customer_id', 'customer_tenure_days']], 
                on='customer_id', 
                how='left'
            )
            
            # Calculate churn probability (simplified model)
            # In production, this would use a trained ML model
            churn_features['churn_probability'] = (
                churn_features['days_since_last_order'] / 365 * 0.3 +
                (1 - churn_features['order_frequency_trend']) * 0.4 +
                (churn_features['avg_order_value'] < churn_features['avg_order_value'].median()) * 0.3
            )
            
            # Cap probability between 0 and 1
            churn_features['churn_probability'] = np.clip(churn_features['churn_probability'], 0, 1)
            
            # Risk categories
            def categorize_risk(prob):
                if prob < 0.2:
                    return 'Low Risk'
                elif prob < 0.5:
                    return 'Medium Risk'
                elif prob < 0.8:
                    return 'High Risk'
                else:
                    return 'Very High Risk'
            
            churn_features['risk_category'] = churn_features['churn_probability'].apply(categorize_risk)
            
            self.logger.info("Customer churn prediction completed")
            return churn_features
            
        except Exception as e:
            self.logger.error(f"Churn prediction failed: {str(e)}")
            raise
    
    def generate_customer_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive customer insights"""
        try:
            self.logger.info("Generating customer insights")
            
            insights = {}
            
            # Basic customer metrics
            insights['total_customers'] = df['customer_id'].nunique()
            insights['total_revenue'] = df['total_amount'].sum()
            insights['avg_order_value'] = df['total_amount'].mean()
            insights['avg_orders_per_customer'] = df.groupby('customer_id')['order_id'].nunique().mean()
            
            # Customer distribution by segments
            if 'customer_segment' in df.columns:
                segment_distribution = df['customer_segment'].value_counts().to_dict()
                insights['segment_distribution'] = segment_distribution
            
            # Top customers by revenue
            top_customers = df.groupby('customer_id')['total_amount'].sum().sort_values(ascending=False).head(10)
            insights['top_customers'] = top_customers.to_dict()
            
            # Customer retention analysis
            if 'order_date' in df.columns:
                # Calculate monthly active customers
                df['order_month'] = df['order_date'].dt.to_period('M')
                monthly_customers = df.groupby('order_month')['customer_id'].nunique()
                insights['monthly_active_customers'] = monthly_customers.to_dict()
            
            # Revenue trends
            if 'order_date' in df.columns:
                df['order_month'] = df['order_date'].dt.to_period('M')
                monthly_revenue = df.groupby('order_month')['total_amount'].sum()
                insights['monthly_revenue'] = monthly_revenue.to_dict()
            
            # Customer lifetime analysis
            customer_lifetime = df.groupby('customer_id').agg({
                'order_date': ['min', 'max'],
                'total_amount': 'sum'
            })
            customer_lifetime.columns = ['first_order', 'last_order', 'total_spent']
            customer_lifetime['lifetime_days'] = (
                customer_lifetime['last_order'] - customer_lifetime['first_order']
            ).dt.days
            
            insights['avg_customer_lifetime_days'] = customer_lifetime['lifetime_days'].mean()
            insights['avg_customer_lifetime_value'] = customer_lifetime['total_spent'].mean()
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Customer insights generation failed: {str(e)}")
            return {}
    
    def run_complete_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete customer analytics pipeline"""
        try:
            self.logger.info("Starting complete customer analytics pipeline")
            
            results = {}
            
            # RFM Analysis
            rfm_df = self.calculate_rfm_metrics(df)
            results['rfm_analysis'] = rfm_df
            
            # Customer Lifetime Value
            clv_df = self.calculate_customer_lifetime_value(df)
            results['clv_analysis'] = clv_df
            
            # Merge RFM and CLV data
            customer_analytics_df = rfm_df.merge(clv_df, on='customer_id', how='inner')
            
            # Customer Segmentation
            segmentation_results = self.customer_segmentation_clustering(customer_analytics_df)
            results['segmentation'] = segmentation_results
            
            # Anomaly Detection
            anomaly_df = self.detect_anomalies(customer_analytics_df)
            results['anomaly_detection'] = anomaly_df
            
            # Churn Prediction
            churn_df = self.customer_churn_prediction(df)
            results['churn_prediction'] = churn_df
            
            # Generate Insights
            insights = self.generate_customer_insights(df)
            results['insights'] = insights
            
            self.logger.info("Customer analytics pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Customer analytics pipeline failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_orders = 50000
    
    sample_data = {
        'customer_id': [f'CUST_{np.random.randint(1, 1001):06d}' for _ in range(n_orders)],
        'order_id': [f'ORDER_{i:08d}' for i in range(1, n_orders + 1)],
        'product_id': [f'PROD_{np.random.randint(1, 501):06d}' for _ in range(n_orders)],
        'order_date': pd.date_range('2023-01-01', '2024-01-01', periods=n_orders),
        'total_amount': np.random.uniform(10, 1000, n_orders),
        'quantity': np.random.randint(1, 10, n_orders)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analytics
    config = {'random_state': 42}
    analytics = CustomerAnalytics(config)
    
    # Run complete analysis
    results = analytics.run_complete_analysis(df)
    
    print("Customer Analytics completed successfully!")
    print(f"Total customers analyzed: {len(results['rfm_analysis'])}")
    print(f"Customer segments: {results['rfm_analysis']['customer_segment'].value_counts().to_dict()}")
