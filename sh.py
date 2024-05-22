import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from joblib import parallel_backend
from tqdm import tqdm


# [TODO] 결측값을 평균값으로 대체

train_data = pd.read_csv('train.csv')
validation_data = pd.read_csv('validation.csv')
test_data = pd.read_csv('test.csv')
train_data.fillna(train_data.mean(), inplace=True)
validation_data.fillna(validation_data.mean(), inplace=True)

if train_data['sleep_stage'].dtype != 'int':
    train_data['sleep_stage'] = train_data['sleep_stage'].astype(int)
if validation_data['sleep_stage'].dtype != 'int':
    validation_data['sleep_stage'] = validation_data['sleep_stage'].astype(int)

X_train = train_data.drop(columns=['sleep_stage', 'id'])
y_train = train_data['sleep_stage']

X_val = validation_data.drop(columns=['sleep_stage', 'id'])
y_val = validation_data['sleep_stage']


model = RandomForestClassifier()


with parallel_backend('threading', n_jobs=-1):
    for _ in tqdm(range(1)):
        model.fit(X_train, y_train)


y_val_pred = model.predict(X_val)

f1 = f1_score(y_val, y_val_pred, average='weighted')
print(f'Validation F1 Score: {f1:.2f}')


X_test = test_data.drop(columns=['id'])
y_test_pred = model.predict(X_test)

test_data['predicted_sleep_stage'] = y_test_pred

print(test_data[['id', 'predicted_sleep_stage']])