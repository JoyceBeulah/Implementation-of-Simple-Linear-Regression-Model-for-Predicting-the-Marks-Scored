# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R.Joyce Beulah
RegisterNumber:  212222230058
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()

#segregating data to variables
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![image](https://github.com/JoyceBeulah/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343698/4087ba13-0dec-4c34-9739-96d69d8143c1)

Array value of X

![image](https://github.com/JoyceBeulah/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343698/5574a610-d862-4934-9008-339d0b24d640)

Array value of Y

![image](https://github.com/JoyceBeulah/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343698/baf22336-20f7-4ea5-89b8-7a99154b4b0f)

Values of Y prediction

![image](https://github.com/JoyceBeulah/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343698/123e8119-a7fd-4b81-9175-ac83c97433cd)

Values of Y test

![image](https://github.com/JoyceBeulah/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343698/6dce484c-73d1-4862-9d3d-834d60a93dc4)

Training Set Graph
![image](https://github.com/JoyceBeulah/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343698/5e9d5763-52c2-4182-99cf-be2d9eeb1f93)

Test Set Graph
![image](https://github.com/JoyceBeulah/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343698/8dc42749-b792-4694-9c91-096a21b9cd7c)

Values of MSE, MAE and RMSE

![image](https://github.com/JoyceBeulah/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343698/9a97f8c7-101a-477d-8c14-aa0b772b8c7c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
