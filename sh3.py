import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from joblib import parallel_backend
from tqdm import tqdm

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

# 입력 변수(X)와 타겟 변수(y) 분리
X = train_data[['pulse']]
y = train_data['sleep_stage']

# K-fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

# tqdm을 사용하여 K-fold 교차 검증 진행 상황 표시
with parallel_backend('threading', n_jobs=-1):
    for train_index, val_index in tqdm(kf.split(X), desc="K-fold Cross Validation"):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # SVM 모델 학습
        model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
        model.fit(X_train, y_train)

        # 검증 데이터 예측
        y_pred_val = model.predict(X_val)

        # 검증 데이터 f1 score 계산
        f1 = f1_score(y_val, y_pred_val, average='weighted')
        f1_scores.append(f1)

# 교차 검증 결과 출력
average_f1_score = sum(f1_scores) / len(f1_scores)
print(f'평균 검증 데이터 f1 score: {average_f1_score:.2f}')

# 전체 학습 데이터로 모델 학습
model.fit(X, y)

# 검증 데이터 예측
X_validation = validation_data[['pulse']]
y_validation = validation_data['sleep_stage']
y_pred_validation = model.predict(X_validation)

# 검증 데이터 f1 score 출력
validation_f1_score = f1_score(y_validation, y_pred_validation, average='weighted')
print(f'검증 데이터 f1 score: {validation_f1_score:.2f}')

# 테스트 데이터 예측
X_test = test_data[['pulse']]
y_pred_test = model.predict(X_test)

# 예측 결과를 test 데이터에 추가
test_data['predicted_sleep_stage'] = y_pred_test

# 예측 결과 출력
print(test_data[['pulse', 'predicted_sleep_stage']])

# 예측 결과 저장
test_data.to_csv('predicted_test_data.csv', index=False)
