import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

data  = pd.read_csv("diabetes_dataset.csv")
x = data.drop('Outcome',axis=1)
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3)

# decition tree code
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
#print(model.predict([[10,139,80,0,0,27.1,1.441,57]]))
with open('classifier.pkl','wb') as file:
    pickle.dump(model,file)
