# обучение модели
def fit_model():
  # Прочитайте файл с гиперпараметрами params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd) 
    index_col = params['index_col']
    target_col= params['target_col']
    one_hot_drop= params['one_hot_drop']
    auto_class_weights= params['auto_class_weights']
    n_splits= params['n_splits']
    metrics= params['metrics']
    n_jobs= params['n_jobs']   

  # загрузите результат предыдущего шага: inital_data.csv
    data = pd.read_csv('data/initial_data.csv')

  # реализуйте основную логику шага с использованием гиперпараметров
    # обучение модели
    cat_features = data.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(['float'])

    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop=one_hot_drop), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    model = CatBoostClassifier(auto_class_weights=auto_class_weights)

    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    pipeline.fit(data, data[target_col]) 

  # сохраните обученную модель в models/fitted_model.pkl
    os.makedirs('models', exist_ok=True) # создание директории, если её ещё нет
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd)
