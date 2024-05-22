import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_score
from joblib import parallel_backend
from tqdm import tqdm

# [ToDo] Stratified K-Fold Cross-Validation

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

# K-겹 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=None)
model = RandomForestClassifier()

# 교차 검증 수행 및 f1-score 평가
f1_scores = []

with parallel_backend('threading', n_jobs=-1):
    for train_index, val_index in tqdm(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, average='weighted')
        f1_scores.append(f1)

# f1-score의 평균값 출력
print(f'Cross-Validation F1 Score: {sum(f1_scores) / len(f1_scores):.2f}')

# 전체 train_data로 모델 재학습
model.fit(X_train, y_train)

# test 데이터를 사용하여 예측
X_test = test_data.drop(columns=['id'])
y_test_pred = model.predict(X_test)

# 예측 결과를 test_data에 추가
test_data['predicted_sleep_stage'] = y_test_pred

# 예측 결과 출력
print(test_data[['id', 'predicted_sleep_stage']])
