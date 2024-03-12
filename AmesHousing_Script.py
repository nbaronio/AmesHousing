import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# import the dataset and show the first 4 rows
df = pd.read_csv("AmesHousing.csv")
df.head()
print(df.shape)

df.info()

# 2. Data Preparation

# needed to display all rows and cols
pd.set_option('display.max_columns',None) 
pd.set_option('display.max_rows', None) 

df.isna().sum()
df.fillna(value=0, axis=1, inplace=True)

# now we can split our dataframe in two: X will be our features, Y our target variable
X = df.drop("SalePrice", axis=1) # features
Y = df[["SalePrice"]] # Target variable

# 3. Encoding our categorical data

label_encoder = LabelEncoder()
X_categorical = df.select_dtypes(include=["object"]).apply(label_encoder.fit_transform)
X_numerical = df.select_dtypes(exclude=["object"]).values
x = pd.concat([pd.DataFrame(X_numerical), X_categorical], axis=1).values

# 4. Model Training

# split the dataset in two, train and split
from sklearn.model_selection import train_test_split

x_train, x_test, Y_train, Y_test = train_test_split(x,Y, test_size=0.2, random_state=0)

# fitting Random Forest Regression model to the dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

# in this case we will have 10 decision trees
# the oob (out of bag) estimates the model's generalization performance

# fit the regressor with our data
regressor.fit(x_train,Y_train)

# 5. Model Predictions

# making predictions
predictions = regressor.predict(x_test)
Y_test.head(10)

# comparing the Y-test and predictions of our decision tree

comparison = pd.DataFrame({'Actual': Y_test.values.flatten(), 'Predicted': predictions.flatten()})
print(comparison.head(20))

# computing MSE and R2

from sklearn.metrics import mean_squared_error, r2_score

oob_score = round(regressor.oob_score_, 5)
MSE = mean_squared_error(Y_test,predictions)
R2 = r2_score(Y_test,predictions)

print(f"Out Of Bag Score (OOF): {oob_score}") 
print(f"Mean Square Error (MSE): {MSE}") 
print(f"R2: {R2}") 

# Variable importance

# let's see the importance of the various variables that we encountered during our analysis.

# get importance
importance = regressor.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()



