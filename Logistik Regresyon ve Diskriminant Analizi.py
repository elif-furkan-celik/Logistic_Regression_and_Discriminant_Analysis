from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv('student-por.csv')
df = df.sample(frac = 1)
df = df.reset_index()
df = df.fillna(0)
df = df.drop('index', axis=1)

encoder = LabelEncoder()

for i, j in enumerate(df.columns) :
    encoded = encoder.fit_transform(df.iloc[:, i])
    df[j] = encoded

G = []

for i in range(len(df)):
    G.append(df['G1'][i] + df['G2'][i] + df['G3'][i])

labels = G
df = df.drop('G1', axis=1)
df = df.drop('G2', axis=1)
df = df.drop('G3', axis=1)

labels = np.array(labels)

features = df.iloc[:, 0:].values

for i in range(len(df.columns)):
    features[:,i] = list(map(lambda x: ((x-min(features[:,i])) / (max(features[:,i]) - min(features[:,i]))) , features[:,i]))

features = np.array(features)
print("Sample Count :", features.shape)

x_train = features[:int((labels.shape[0] * 80) / 100), :]
y_train = labels[:int((labels.shape[0] * 80) / 100)]
x_test = features[int((labels.shape[0] * 80) / 100):, :]
y_test = labels[int((labels.shape[0] * 80) / 100):]


clf1 = LogisticRegression(penalty='l2').fit(x_train, y_train)
y_pred = []
y_pred.append([clf1.predict(x_test), 'clf1', clf1])

print('\n\nMAPE : %', np.mean(np.abs((y_test - y_pred[0][0]) / y_test)) * 100)
print('MAE  : ', mean_absolute_error(y_test, y_pred[0][0]))
print('MSE  : ', mean_squared_error(y_test, y_pred[0][0], squared=False))


clf2 = LogisticRegression(solver='liblinear', penalty='l1').fit(x_train, y_train)
y_pred.append([clf2.predict(x_test), 'clf2', clf2])

print('\n\nMAPE : %', np.mean(np.abs((y_test - y_pred[1][0]) / y_test)) * 100)
print('MAE  : ', mean_absolute_error(y_test, y_pred[1][0]))
print('MSE  : ', mean_squared_error(y_test, y_pred[1][0], squared=False))


clf3 = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.).fit(x_train, y_train)
y_pred.append([clf3.predict(x_test), 'clf3', clf3])

print('\n\nMAPE : %', np.mean(np.abs((y_test - y_pred[2][0]) / y_test)) * 100)
print('MAE  : ', mean_absolute_error(y_test, y_pred[2][0]))
print('MSE  : ', mean_squared_error(y_test, y_pred[2][0], squared=False))


clf4 = LogisticRegression(solver='newton-cg', penalty='none').fit(x_train, y_train)
y_pred.append([clf4.predict(x_test), 'clf4', clf4])

print('\n\nMAPE : %', np.mean(np.abs((y_test - y_pred[3][0]) / y_test)) * 100)
print('MAE  : ', mean_absolute_error(y_test, y_pred[3][0]))
print('MSE  : ', mean_squared_error(y_test, y_pred[3][0], squared=False))


lda1 = LinearDiscriminantAnalysis(solver='svd')
lda1.fit(x_train, y_train)

y_pred.append([lda1.predict(x_test), 'lda1', lda1])

print('\n\nMAPE : %', np.mean(np.abs((y_test - y_pred[4][0]) / y_test)) * 100)
print('MAE  : ', mean_absolute_error(y_test, y_pred[4][0]))
print('MSE  : ', mean_squared_error(y_test, y_pred[4][0], squared=False))


lda2 = LinearDiscriminantAnalysis(solver='lsqr')
lda2.fit(x_train, y_train)

y_pred.append([lda2.predict(x_test), 'lda2', lda2])

print('\n\nMAPE : %', np.mean(np.abs((y_test - y_pred[5][0]) / y_test)) * 100)
print('MAE  : ', mean_absolute_error(y_test, y_pred[5][0]))
print('MSE  : ', mean_squared_error(y_test, y_pred[5][0], squared=False))


lda3 = LinearDiscriminantAnalysis(solver='eigen')
lda3.fit(x_train, y_train)

y_pred.append([lda3.predict(x_test), 'lda3', lda3])

print('\n\nMAPE : %', np.mean(np.abs((y_test - y_pred[6][0]) / y_test)) * 100)
print('MAE  : ', mean_absolute_error(y_test, y_pred[6][0]))
print('MSE  : ', mean_squared_error(y_test, y_pred[6][0], squared=False))


best_pred = y_pred[0]
for i in y_pred:
    if  mean_absolute_error(y_test, i[0]) <= mean_absolute_error(y_test, best_pred[0]):
        best_pred = i

print('\n\nModel        : ', best_pred[1])
print('Model Detail : ', best_pred[2])
print('MAPE         : %', np.mean(np.abs((y_test - best_pred[0]) / y_test)) * 100)
print('MAE          : ', mean_absolute_error(y_test, best_pred[0]))
print('MSE          : ', mean_squared_error(y_test, best_pred[0], squared=False))