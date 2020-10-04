
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("/content/wildfires.csv")
df = dataset.drop(columns = ["FIRE_YEAR", "DISCOVERY_DOY", "City", "datetime"])
df = df.sample(frac=1).reset_index(drop=True)
df.head(100000)

y = df.FIRE_SIZE.values
X = df.drop(columns = ["FIRE_SIZE"]).values

normalized_X = preprocessing.normalize(X)

plt.plot(normalized_X, y)
plt.xlabel('X')
plt.ylabel('y')

#separating into classes
for i in range(len(y)):
  if y[i]<10000:
    y[i]=0
  elif y[i]>=10000 and y[i]<50000:
    y[i]=1
  elif y[i]>=50000 and y[i]<100000:
    y[i]=2
  elif y[i]>=100000 and y[i]<150000:
    y[i]=3
  elif y[i]>=150000 and y[i]<200000:
    y[i]=4
  elif y[i]>=20000:
    y[i]=5

labels=np.ndarray(shape=(len(y),1),
                      dtype=np.int64)
k = 0
for i in y:
  labels[k]=(int(i))
  k += 1

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

Y = convert_to_one_hot(labels, 6).T

X_train, X_test, y_train, y_test = train_test_split(normalized_X, labels, test_size = 0.2, random_state = 42)

y_train_orig=y_train.T
y_test_orig=y_test.T

Y_train = convert_to_one_hot(y_train_orig, 6).T
Y_test = convert_to_one_hot(y_test_orig, 6).T

from sklearn.ensemble import ExtraTreesRegressor
reg = ExtraTreesRegressor(n_estimators=100, random_state=0)
reg.fit(X_train, Y_train)
reg.score(X_test, Y_test)
  
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_pred, Y_test)

mse

filename = 'firesize_model.sav'
pickle.dump(reg, open(filename, "wb"))

model = pickle.load(open('/content/firesize_model.sav', 'rb'))

Xnew=X_test[500:501]

ynew=np.argmax(reg.predict(Xnew))

if ynew ==0:
  print("Estimated Fire Size is less than 10000")
elif ynew ==1:
  print("Estimated Fire Size is between 10000 and 50000")
elif ynew ==2:
  print("Estimated Fire Size is between 50000 and 100000")
elif ynew ==3:
  print("Estimated Fire Size is between 100000 and 150000")
elif ynew ==4:
  print("Estimated Fire Size is between 150000 and 200000")
elif ynew ==5:
  print("Estimated Fire Size is greater than 200000")

