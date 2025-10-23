"""
Advanced ML Models for Sales Forecasting
Comprehensive machine learning pipeline for sales prediction and analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
import json
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import optuna
from optuna.integration import LightGBMPruningCallback

class SalesForecastingPipeline:
    """Advanced sales forecasting pipeline with multiple ML models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for ML models"""
        try:
            self.logger.info("Preparing features for ML models")
            
            # Create a copy to avoid modifying original data
            features_df = df.copy()
            
            # Date features
            if 'order_date' in features_df.columns:
                features_df['order_date'] = pd.to_datetime(features_df['order_date'])
                features_df['year'] = features_df['order_date'].dt.year
                features_df['month'] = features_df['order_date'].dt.month
                features_df['day'] = features_df['order_date'].dt.day
                features_df['day_of_week'] = features_df['order_date'].dt.dayofweek
                features_df['day_of_year'] = features_df['order_date'].dt.dayofyear
                features_df['week_of_year'] = features_df['order_date'].dt.isocalendar().week
                features_df['quarter'] = features_df['order_date'].dt.quarter
                features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
                features_df['is_month_start'] = features_df['order_date'].dt.is_month_start.astype(int)
                features_df['is_month_end'] = features_df['order_date'].dt.is_month_end.astype(int)
            
            # Lag features
            if 'total_amount' in features_df.columns:
                features_df = features_df.sort_values(['customer_id', 'order_date'])
                features_df['amount_lag_1'] = features_df.groupby('customer_id')['total_amount'].shift(1)
                features_df['amount_lag_7'] = features_df.groupby('customer_id')['total_amount'].shift(7)
                features_df['amount_lag_30'] = features_df.groupby('customer_id')['total_amount'].shift(30)
                
                # Rolling statistics
                features_df['amount_rolling_7'] = features_df.groupby('customer_id')['total_amount'].rolling(7).mean().reset_index(0, drop=True)
                features_df['amount_rolling_30'] = features_df.groupby('customer_id')['total_amount'].rolling(30).mean().reset_index(0, drop=True)
                features_df['amount_rolling_90'] = features_df.groupby('customer_id')['total_amount'].rolling(90).mean().reset_index(0, drop=True)
                
                # Rolling standard deviation
                features_df['amount_std_7'] = features_df.groupby('customer_id')['total_amount'].rolling(7).std().reset_index(0, drop=True)
                features_df['amount_std_30'] = features_df.groupby('customer_id')['total_amount'].rolling(30).std().reset_index(0, drop=True)
            
            # Customer features
            if 'customer_id' in features_df.columns:
                customer_stats = features_df.groupby('customer_id').agg({
                    'total_amount': ['mean', 'std', 'min', 'max', 'count'],
                    'quantity': ['mean', 'std', 'sum']
                }).reset_index()
                
                customer_stats.columns = ['customer_id', 'customer_avg_amount', 'customer_std_amount', 
                                        'customer_min_amount', 'customer_max_amount', 'customer_order_count',
                                        'customer_avg_quantity', 'customer_std_quantity', 'customer_total_quantity']
                
                features_df = features_df.merge(customer_stats, on='customer_id', how='left')
            
            # Product features
            if 'product_id' in features_df.columns:
                product_stats = features_df.groupby('product_id').agg({
                    'total_amount': ['mean', 'std', 'count'],
                    'quantity': ['mean', 'sum']
                }).reset_index()
                
                product_stats.columns = ['product_id', 'product_avg_amount', 'product_std_amount', 
                                       'product_order_count', 'product_avg_quantity', 'product_total_quantity']
                
                features_df = features_df.merge(product_stats, on='product_id', how='left')
            
            # Category features
            if 'category' in features_df.columns:
                category_stats = features_df.groupby('category').agg({
                    'total_amount': ['mean', 'std', 'count']
                }).reset_index()
                
                category_stats.columns = ['category', 'category_avg_amount', 'category_std_amount', 'category_order_count']
                features_df = features_df.merge(category_stats, on='category', how='left')
            
            # Time-based features
            if 'order_date' in features_df.columns:
                # Days since last order
                features_df = features_df.sort_values(['customer_id', 'order_date'])
                features_df['days_since_last_order'] = features_df.groupby('customer_id')['order_date'].diff().dt.days
                
                # Days since first order
                first_order = features_df.groupby('customer_id')['order_date'].min().reset_index()
                first_order.columns = ['customer_id', 'first_order_date']
                features_df = features_df.merge(first_order, on='customer_id', how='left')
                features_df['days_since_first_order'] = (features_df['order_date'] - features_df['first_order_date']).dt.days
            
            # Seasonal features
            if 'month' in features_df.columns:
                features_df['sin_month'] = np.sin(2 * np.pi * features_df['month'] / 12)
                features_df['cos_month'] = np.cos(2 * np.pi * features_df['month'] / 12)
                features_df['sin_day'] = np.sin(2 * np.pi * features_df['day'] / 31)
                features_df['cos_day'] = np.cos(2 * np.pi * features_df['day'] / 31)
            
            # Handle missing values
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            features_df[numeric_columns] = features_df[numeric_columns].fillna(features_df[numeric_columns].mean())
            
            self.logger.info(f"Feature engineering completed. Shape: {features_df.shape}")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {str(e)}")
            raise
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train multiple ML models for sales forecasting"""
        try:
            self.logger.info("Training ML models for sales forecasting")
            
            # Prepare feature lists
            numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
            
            # Encode categorical features
            for feature in categorical_features:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    X_train[feature] = self.encoders[feature].fit_transform(X_train[feature].astype(str))
                    X_val[feature] = self.encoders[feature].transform(X_val[feature].astype(str))
                else:
                    X_train[feature] = self.encoders[feature].transform(X_train[feature].astype(str))
                    X_val[feature] = self.encoders[feature].transform(X_val[feature].astype(str))
            
            # Scale features
            self.scalers['standard'] = StandardScaler()
            X_train_scaled = self.scalers['standard'].fit_transform(X_train[numeric_features])
            X_val_scaled = self.scalers['standard'].transform(X_val[numeric_features])
            
            # Convert back to DataFrame
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_features, index=X_train.index)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=numeric_features, index=X_val.index)
            
            # Combine scaled numeric and encoded categorical features
            X_train_final = pd.concat([X_train_scaled, X_train[categorical_features]], axis=1)
            X_val_final = pd.concat([X_val_scaled, X_val[categorical_features]], axis=1)
            
            # Train models
            models_to_train = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            }
            
            model_performance = {}
            
            for name, model in models_to_train.items():
                self.logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train_final, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_final)
                y_pred_val = model.predict(X_val_final)
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, y_pred_train)
                val_mae = mean_absolute_error(y_val, y_pred_val)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                train_r2 = r2_score(y_train, y_pred_train)
                val_r2 = r2_score(y_val, y_pred_val)
                
                # Store model and performance
                self.models[name] = model
                model_performance[name] = {
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_r2': train_r2,
                    'val_r2': val_r2
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X_train_final.columns, model.feature_importances_))
                
                self.logger.info(f"{name} - Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}")
            
            self.model_performance = model_performance
            return model_performance
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Hyperparameter tuning using Optuna"""
        try:
            self.logger.info("Starting hyperparameter tuning with Optuna")
            
            def objective(trial):
                # XGBoost parameters
                xgb_params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42
                }
                
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                
                return mae
            
            # Run optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)
            
            # Train best model
            best_params = study.best_params
            best_model = xgb.XGBRegressor(**best_params, random_state=42)
            best_model.fit(X_train, y_train)
            
            # Evaluate best model
            y_pred_train = best_model.predict(X_train)
            y_pred_val = best_model.predict(X_val)
            
            best_performance = {
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'val_mae': mean_absolute_error(y_val, y_pred_val),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'train_r2': r2_score(y_train, y_pred_train),
                'val_r2': r2_score(y_val, y_pred_val),
                'best_params': best_params
            }
            
            self.models['xgboost_tuned'] = best_model
            self.model_performance['xgboost_tuned'] = best_performance
            
            self.logger.info(f"Best XGBoost parameters: {best_params}")
            self.logger.info(f"Best model performance - Val MAE: {best_performance['val_mae']:.4f}")
            
            return best_performance
            
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {str(e)}")
            raise
    
    def create_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Create and train LSTM model for time series forecasting"""
        try:
            self.logger.info("Creating LSTM model for time series forecasting")
            
            # Reshape data for LSTM (samples, timesteps, features)
            if len(X_train.shape) == 2:
                X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_val_lstm = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
            else:
                X_train_lstm = X_train
                X_val_lstm = X_val
            
            # Create LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_train_lstm, y_train,
                validation_data=(X_val_lstm, y_val),
                epochs=100,
                batch_size=32,
                verbose=0
            )
            
            # Evaluate model
            y_pred_train = model.predict(X_train_lstm).flatten()
            y_pred_val = model.predict(X_val_lstm).flatten()
            
            lstm_performance = {
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'val_mae': mean_absolute_error(y_val, y_pred_val),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'train_r2': r2_score(y_train, y_pred_train),
                'val_r2': r2_score(y_val, y_pred_val)
            }
            
            self.models['lstm'] = model
            self.model_performance['lstm'] = lstm_performance
            
            self.logger.info(f"LSTM model performance - Val MAE: {lstm_performance['val_mae']:.4f}")
            
            return model, lstm_performance
            
        except Exception as e:
            self.logger.error(f"LSTM model creation failed: {str(e)}")
            raise
    
    def ensemble_prediction(self, X: pd.DataFrame) -> np.ndarray:
        """Create ensemble prediction from multiple models"""
        try:
            predictions = []
            weights = []
            
            # Get predictions from all models
            for name, model in self.models.items():
                if name == 'lstm':
                    # Handle LSTM differently
                    X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
                    pred = model.predict(X_lstm).flatten()
                else:
                    pred = model.predict(X)
                
                predictions.append(pred)
                
                # Weight based on model performance
                if name in self.model_performance:
                    weight = 1 / (1 + self.model_performance[name]['val_mae'])
                    weights.append(weight)
                else:
                    weights.append(1.0)
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Calculate weighted ensemble
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {str(e)}")
            raise
    
    def save_models(self, model_path: str):
        """Save trained models and preprocessing objects"""
        try:
            model_path = Path(model_path)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                if name == 'lstm':
                    model.save(model_path / f"{name}_model.h5")
                else:
                    joblib.dump(model, model_path / f"{name}_model.pkl")
            
            # Save scalers and encoders
            joblib.dump(self.scalers, model_path / "scalers.pkl")
            joblib.dump(self.encoders, model_path / "encoders.pkl")
            
            # Save performance metrics
            with open(model_path / "model_performance.json", 'w') as f:
                json.dump(self.model_performance, f, indent=2)
            
            # Save feature importance
            with open(model_path / "feature_importance.json", 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            self.logger.info(f"Models saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_models(self, model_path: str):
        """Load trained models and preprocessing objects"""
        try:
            model_path = Path(model_path)
            
            # Load scalers and encoders
            self.scalers = joblib.load(model_path / "scalers.pkl")
            self.encoders = joblib.load(model_path / "encoders.pkl")
            
            # Load performance metrics
            with open(model_path / "model_performance.json", 'r') as f:
                self.model_performance = json.load(f)
            
            # Load feature importance
            with open(model_path / "feature_importance.json", 'r') as f:
                self.feature_importance = json.load(f)
            
            # Load models
            model_files = list(model_path.glob("*_model.*"))
            for model_file in model_files:
                name = model_file.stem.replace("_model", "")
                
                if model_file.suffix == '.h5':
                    from tensorflow.keras.models import load_model
                    self.models[name] = load_model(model_file)
                else:
                    self.models[name] = joblib.load(model_file)
            
            self.logger.info(f"Models loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise
    
    def run_complete_pipeline(self, df: pd.DataFrame, target_column: str = 'total_amount') -> Dict[str, Any]:
        """Run complete ML pipeline"""
        try:
            self.logger.info("Starting complete ML pipeline")
            
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Select features and target
            feature_columns = [col for col in features_df.columns if col != target_column and col != 'order_date']
            X = features_df[feature_columns]
            y = features_df[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Train models
            model_performance = self.train_models(X_train, y_train, X_val, y_val)
            
            # Hyperparameter tuning
            tuned_performance = self.hyperparameter_tuning(X_train, y_train, X_val, y_val)
            
            # Create LSTM model
            lstm_model, lstm_performance = self.create_lstm_model(
                X_train.values, y_train.values, X_val.values, y_val.values
            )
            
            # Final evaluation on test set
            test_predictions = self.ensemble_prediction(X_test)
            test_mae = mean_absolute_error(y_test, test_predictions)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_r2 = r2_score(y_test, test_predictions)
            
            results = {
                'model_performance': model_performance,
                'tuned_performance': tuned_performance,
                'lstm_performance': lstm_performance,
                'test_performance': {
                    'mae': test_mae,
                    'rmse': test_rmse,
                    'r2': test_r2
                },
                'feature_importance': self.feature_importance
            }
            
            self.logger.info("ML pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"ML pipeline failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Load sample data
    np.random.seed(42)
    n_samples = 10000
    
    sample_data = {
        'customer_id': [f'CUST_{i:06d}' for i in range(1, n_samples + 1)],
        'product_id': [f'PROD_{np.random.randint(1, 501):06d}' for _ in range(n_samples)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_samples),
        'order_date': pd.date_range('2023-01-01', '2024-01-01', periods=n_samples),
        'total_amount': np.random.uniform(10, 1000, n_samples),
        'quantity': np.random.randint(1, 10, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize pipeline
    config = {
        'model_path': './models',
        'random_state': 42
    }
    
    pipeline = SalesForecastingPipeline(config)
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(df)
    
    # Save models
    pipeline.save_models('./models')
    
    print("ML Pipeline completed successfully!")
    print(f"Test MAE: {results['test_performance']['mae']:.4f}")
    print(f"Test R²: {results['test_performance']['r2']:.4f}")
