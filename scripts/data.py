## scripts/data.py

import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import yaml

def create_connection():

    load_dotenv()
    host = os.environ.get('DB_DESTINATION_HOST')
    port = os.environ.get('DB_DESTINATION_PORT')
    db = os.environ.get('DB_DESTINATION_NAME')
    username = os.environ.get('DB_DESTINATION_USER')
    password = os.environ.get('DB_DESTINATION_PASSWORD')
    
    print(f'postgresql://{username}:{password}@{host}:{port}/{db}')
    conn = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{db}', connect_args={'sslmode':'require'})
    return conn

def get_data():
    # загрузка гиперпараметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    # загрузки предыдущих результатов нет, так как это первый шаг

    # основная логика
    conn = create_connection()
    data = pd.read_sql('select * from clean_users_churn', conn, index_col=params['index_col'])
    conn.dispose()

    # сохранение результата шага
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/initial_data.csv', index=None)


if __name__ == '__main__':
    get_data()