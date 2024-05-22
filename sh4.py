import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
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

# 모델 정의
rf_model = RandomForestClassifier()
lr_model = LogisticRegression(max_iter=200)
svc_model = SVC(probability=True)

# Voting Classifier 설정
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('lr', lr_model), ('svc', svc_model)],
    voting='soft'  # 소프트 보팅을 사용하여 확률을 기반으로 예측
)

# 모델 학습
with parallel_backend('threading', n_jobs=-1):
    for _ in tqdm(range(1)):
        voting_model.fit(X_train, y_train)

# 검증 데이터에 대해 예측
y_val_pred = voting_model.predict(X_val)

# f1-score 평가
f1 = f1_score(y_val, y_val_pred, average='weighted')
print(f'Validation F1 Score: {f1:.2f}')

# test 데이터를 사용하여 예측
X_test = test_data.drop(columns=['id'])
y_test_pred = voting_model.predict(X_test)

# 예측 결과를 test_data에 추가
test_data['predicted_sleep_stage'] = y_test_pred

# 예측 결과 출력
print(test_data[['id', 'predicted_sleep_stage']])
