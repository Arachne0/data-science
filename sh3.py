import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend
from tqdm import tqdm

# 데이터 불러오기
train_data = pd.read_csv('train.csv')
validation_data = pd.read_csv('validation.csv')
test_data = pd.read_csv('test.csv')

# 결측값을 평균값으로 대체
train_data.fillna(train_data.mean(), inplace=True)
validation_data.fillna(validation_data.mean(), inplace=True)

# sleep_stage가 범주형 값인지 확인하고, 필요시 변환
if train_data['sleep_stage'].dtype != 'int':
    train_data['sleep_stage'] = train_data['sleep_stage'].astype(int)
if validation_data['sleep_stage'].dtype != 'int':
    validation_data['sleep_stage'] = validation_data['sleep_stage'].astype(int)

# feature와 label 분리
X_train = train_data.drop(columns=['sleep_stage', 'id'])
y_train = train_data['sleep_stage']

X_val = validation_data.drop(columns=['sleep_stage', 'id'])
y_val = validation_data['sleep_stage']

# RandomForestClassifier 모델 정의
model = RandomForestClassifier()

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV 설정
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)

# 모델 학습 및 최적화
with parallel_backend('threading', n_jobs=-1):
    grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print(f'Best parameters found: {grid_search.best_params_}')
print(f'Best cross-validation F1 score: {grid_search.best_score_:.2f}')

# 최적 모델로 검증 데이터에 대해 예측
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

# f1-score 평가
f1 = f1_score(y_val, y_val_pred, average='weighted')
print(f'Validation F1 Score with best model: {f1:.2f}')

# test 데이터를 사용하여 예측
X_test = test_data.drop(columns=['id'])
y_test_pred = best_model.predict(X_test)

# 예측 결과를 test_data에 추가
test_data['predicted_sleep_stage'] = y_test_pred

# 예측 결과 출력
print(test_data[['id', 'predicted_sleep_stage']])
