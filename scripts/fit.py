# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression  
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
    #X = data.drop(params['target_col'], axis=1)  # Признаки
    #y = data[params['target_col']]               # Целевая переменная

    # Определение типов признаков
    cat_features = data.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2
    
    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(include=['float', 'int'])  # Добавлены int-признаки

    # Создание препроцессора
    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop=params['one_hot_drop']), 
                binary_cat_features.columns.tolist()),
            ('cat', OneHotEncoder(handle_unknown=params['handle_unknown']), 
                other_cat_features.columns.tolist()),
            ('num', StandardScaler(), 
                num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
 
    # Инициализация модели
    model = LogisticRegression(
        C=params['C'], 
        penalty=params['penalty']#,
        #class_weight=params['class_weight']
    )

    # Создание пайплайна
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Обучение модели
    pipeline.fit(data, data['target'])  # Четкое разделение X и y

    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fitted_model.pkl')  # Правильное сохранение модели

    print("Model successfully trained and saved!")

if __name__ == '__main__':
    fit_model()