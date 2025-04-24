# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
import yaml
import os
import joblib

def fit_model():
    # Загрузка параметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # Загрузка данных
    data = pd.read_csv('data/initial_data.csv')
    
    # Проверка наличия целевой s
    if params['target_col'] not in data.columns:
        raise ValueError(f"Target column '{params['target_col']}' not found in data")

    # Разделение признаков
    X = data.drop(params['target_col'], axis=1)  # Признаки
    y = data[params['target_col']]               # Целевая переменная
    # Определение типов признаков
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Разделение категориальных признаков на бинарные и остальные
    binary_cat_features = [col for col in categorical_features 
                     if X[col].nunique() == 2]
    other_cat_features = [col for col in categorical_features 
                        if col not in binary_cat_features]
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=params['strategy_num'])),
        ('scaler', StandardScaler())
    ])
    
    binary_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=params['strategy'])),
        ('encoder', OneHotEncoder(drop=params['one_hot_drop'], sparse_output=False))
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy=params['strategy'])),
        ('encoder', OneHotEncoder(handle_unknown=params['handle_unknown'], sparse_output=False))
    ])

    # Создание ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_features),
        ('binary', binary_transformer, binary_cat_features),
        ('cat', categorical_transformer, other_cat_features)
    ])

 
    # Инициализация модели
    model = LogisticRegression(
        C=params['C'], 
        penalty=params['penalty'],
        class_weight=params['class_weight']
    )

    # Создание пайплайна
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Обучение модели
    pipeline.fit(X, y)  # Четкое разделение X и y

    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fitted_model.pkl')  # Правильное сохранение модели

    print("Model successfully trained and saved!")

if __name__ == '__main__':
    fit_model()