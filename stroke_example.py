import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
df = df.drop('id',axis = 1)

# checking for null/ missing values
sns.heatmap(df.isnull(), cbar=False )
plt.show()



# replacing null values with mean
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# one hot encoding
categorical_vars = ['gender','ever_married',
       'work_type', 'Residence_type','smoking_status']

encoded_df = pd.get_dummies(data = df,columns = categorical_vars,drop_first = 1)



# visualising the imbalance of data
def count_graph(df):
    count_strokes = pd.value_counts(df['stroke'], sort = True)

    count_strokes.plot(kind = 'bar', rot=0)

    plt.title("stroke graph ")


    plt.xlabel("stroke")

    plt.ylabel("Frequency")
    plt.show()

# count_graph(df)

# one hot encoding

X = encoded_df.drop('stroke',axis = 1)
Y = encoded_df.stroke

from imblearn.combine import SMOTETomek
smk = SMOTETomek()
x_res,y_res = smk.fit_sample(X,Y)

# print(x_res.shape,y_res.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_res,y_res,test_size = 0.25 , random_state = 0)
# print(x_train.shape,x_test.shape)

def logistic_regression(x_train,y_train,x_test,y_test,x_res,y_res):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(x_train,y_train)

    from sklearn.metrics import confusion_matrix
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    print('classification score for knearest_neighbour {} ' .format(classifier.score(x_res,y_res)))
logistic_regression(x_train,y_train,x_test,y_test,x_res,y_res)

def knearest_neighbour(x_train,y_train,x_test,y_test,x_res,y_res):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5 )
    classifier.fit(x_train,y_train)

    from sklearn.metrics import confusion_matrix
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    print('classification score for knearest_neighbour {} ' .format(classifier.score(x_res,y_res))) 
    sns.heatmap(cm, cbar = False, annot = True)
    plt.show()

knearest_neighbour(x_train,y_train,x_test,y_test,x_res,y_res)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               