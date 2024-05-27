import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from joblib import parallel_backend
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
import numpy as np

# 데이터 로드
train_data = pd.read_csv('train.csv')
validation_data = pd.read_csv('validation.csv')
test_data = pd.read_csv('test.csv')

# 결측값을 평균값으로 대체
train_data.fillna(train_data.mean(), inplace=True)
validation_data.fillna(validation_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

if train_data['sleep_stage'].dtype != 'int':
    train_data['sleep_stage'] = train_data['sleep_stage'].astype(int)
if validation_data['sleep_stage'].dtype != 'int':
    validation_data['sleep_stage'] = validation_data['sleep_stage'].astype(int)

# 유효하지 않은 클래스 레이블 확인 및 제거
valid_classes = [0, 1, 2, 3, 4, 5]  # 예를 들어 유효한 클래스 레이블 목록
train_data = train_data[train_data['sleep_stage'].isin(valid_classes)]
validation_data = validation_data[validation_data['sleep_stage'].isin(valid_classes)]

# 입력 변수(X)와 타겟 변수(y) 분리
X = train_data[['pulse']]
y = train_data['sleep_stage']

# 클래스 가중치 계산
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}

# K-fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

# SVM 모델 하이퍼파라미터 그리드 정의
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# tqdm을 사용하여 K-fold 교차 검증 진행 상황 표시
with parallel_backend('threading', n_jobs=-1):
    for train_index, val_index in tqdm(kf.split(X), desc="K-fold Cross Validation"):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # SVM 모델 하이퍼파라미터 튜닝
        svm = SVC(class_weight=class_weights_dict)
        svm_grid = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
        svm_grid.fit(X_train, y_train)

        # 최적의 모델로 예측 수행
        best_model = svm_grid.best_estimator_
        y_pred_val = best_model.predict(X_val)

        # 검증 데이터 f1 score 계산
        f1 = f1_score(y_val, y_pred_val, average='weighted')
        f1_scores.append(f1)

# 교차 검증 결과 출력
average_f1_score = sum(f1_scores) / len(f1_scores)
print(f'평균 검증 데이터 f1 score: {average_f1_score:.2f}')

# 최적의 모델로 전체 학습 데이터로 모델 학습
best_model.fit(X, y)

# 검증 데이터 예측
X_validation = validation_data[['pulse']]
y_validation = validation_data['sleep_stage']
y_pred_validation = best_model.predict(X_validation)

# 검증 데이터 f1 score 출력
validation_f1_score = f1_score(y_validation, y_pred_validation, average='weighted')
print(f'검증 데이터 f1 score: {validation_f1_score:.2f}')

# 테스트 데이터 예측
X_test = test_data[['pulse']]
y_pred_test = best_model.predict(X_test)

# 예측 결과를 test 데이터에 추가
test_data['predicted_sleep_stage'] = y_pred_test

# 예측 결과 출력
print(test_data[['pulse', 'predicted_sleep_stage']])

# 예측 결과 저장
test_data.to_csv('predicted_test_data.csv', index=False)
