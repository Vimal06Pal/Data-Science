import pandas as pd
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


ipl_df = pd.read_csv('./ipl.csv')
# print(ipl_df)

# print(ipl_df.dtypes)

# data cleaning 
# removing unwanted columns
print(ipl_df.columns)
def remove_unwanted_cols(ipl_df):
    columns_to_remove = ['mid','venue','batsman','bowler','striker','non-striker']
    ipl_df.drop(labels = columns_to_remove,axis = 1, inplace = True)
    return(ipl_df)
df = remove_unwanted_cols(ipl_df)
# print(df.shape)

# print(df['bat_team'].unique())
# print(df['bowl_team'].unique())

def containg_consisting_team(df):
    consisting_team = ['Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals',
                    'Mumbai Indians' ,'Kings XI Punjab', 'Royal Challengers Bangalore' ,
                    'Delhi Daredevils','Sunrisers Hyderabad' ]
    df = df[(df['bat_team'].isin(consisting_team)) & (df['bowl_team'].isin(consisting_team))]
    return(df)

consistent_df = containg_consisting_team(df)
# print(consistent_df['ball_team'].unique())
# print(consistent_df['bat_team'].unique())

# removing first 5 overs because only after that we can predict

def remove_5_overs(consistent_df):
    consistent_df=consistent_df[consistent_df['overs']>=5.0]
    return(consistent_df)
df= remove_5_overs(consistent_df)
# print(df)

# Converting the column 'date' from string into datetime object
def coverting_date_time(df):
    print("Before converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    # print("After converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    return(df)

df = coverting_date_time(df)
# print(type(df.iloc[0,0]))


# Get correlation of all the features of the dataset
def plotting_heatmap(df):
    corr_matrix = df.corr()
    top_corr_features = corr_matrix.index
    plt.figure(figsize = (13,10))
    sns.heatmap(data =df[top_corr_features].corr(),annot = True,cmap = 'YlGnBu')
    plt.show()

plotting_heatmap(df)


# Data Preprocessing

def onehot_encoding(df):
    encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
    return(encoded_df)

df = onehot_encoding(df)
# print(df.columns)

#rearranging_cols

def rearranging_cols(df):
    df = df[['date','bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
    return(df)

df = rearranging_cols(df)
print("after re-arranging df become {}".format(df.head))

# Splitting the data into train and test set

def splitting_train_test(df):
    x_train = df.drop(labels = 'total',axis =1)[df['date'].dt.year <=2016]
    x_test =  df.drop(labels = 'total',axis=1)[df['date'].dt.year >= 2017]
    # print(x_test.shape)
    y_train = df[df['date'].dt.year <= 2016]['total'].values
    y_test = df[df['date'].dt.year >= 2017]['total'].values

    # Removing the 'date' column
    x_train.drop(labels='date', axis=True, inplace=True)
    x_test.drop(labels='date', axis=True, inplace=True)
    return(x_train,x_test,y_train,y_test)

x_train,x_test,y_train,y_test = splitting_train_test(df)

print("Training set: {} and Test set: {}".format(x_train.shape, x_test.shape))

# Linear Regression
def linear_regressor_model(x_train,y_train,x_test):
    from sklearn.linear_model import LinearRegression
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train,y_train)
    # predicting result
    y_pred = linear_regressor.predict(x_test)
    return(y_pred)

y_pred=linear_regressor_model(x_train,y_train,x_test)



# model Evaluation --------> Linear regression 
def model_evaluation(y_test,y_pred):
    from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,accuracy_score
    print("---- Linear Regression - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred))))

model_evaluation(y_test,y_pred)