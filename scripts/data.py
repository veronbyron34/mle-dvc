## scripts/data.py

# 1 — импорты
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import yaml
#ddd
# 2 — вспомогательные функции
def create_connection():

    load_dotenv()
    host = os.environ.get('DB_DESTINATION_HOST')
    port = os.environ.get('DB_DESTINATION_PORT')
    db = os.environ.get('DB_DESTINATION_NAME')
    username = os.environ.get('DB_DESTINATION_USER')
    password = os.environ.get('DB_DESTINATION_PASSWORD')
    
    print(f'postgresql://mle_20250327_5578ab52f6:b9b985c9dcf248f789895358d315f227@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250327_5578ab52f6')
    conn = create_engine(f'postgresql://mle_20250327_5578ab52f6:b9b985c9dcf248f789895358d315f227@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250327_5578ab52f6')
    return conn

# 3 — главная функция
def get_data():

    # 3.1 — загрузка гиперпараметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)


    # 3.3 — основная логика
    conn = create_connection()
    data = pd.read_sql('select * from clean_users_churn', conn, index_col=params['index_col'])
    conn.dispose()

    # 3.4 — сохранение результата шага
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/initial_data.csv', index=None)

# 4 — защищённый вызов главной функции
if __name__ == '__main__':
    get_data()
