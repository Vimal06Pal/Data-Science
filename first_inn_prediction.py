import pandas as pd
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


ipl_df = pd.read_csv('./ipl.csv')
# print(ipl_df)

# print(ipl_df.dtypes)

''' data cleaning ''' 
# removing unwanted columns
print(ipl_df.columns)
def remove_unwanted_cols(ipl_df):
    columns_to_remove = ['mid','venue','batsman','bowler','striker','non-striker']
    ipl_df.drop(labels = columns_to_remove,axis = 1, inplace = True)
    return(ipl_df)
# print(df.shape)

# print(df['bat_team'].unique())
# print(df['bowl_team'].unique())

def containg_consisting_team(df):
    consisting_team = ['Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals',
                    'Mumbai Indians' ,'Kings XI Punjab', 'Royal Challengers Bangalore' ,
                    'Delhi Daredevils','Sunrisers Hyderabad' ]
    df = df[(df['bat_team'].isin(consisting_team)) & (df['bowl_team'].isin(consisting_team))]
    return(df)

# print(consistent_df['ball_team'].unique())
# print(consistent_df['bat_team'].unique())

# removing first 5 overs because only after that we can predict

def remove_5_overs(consistent_df):
    consistent_df=consistent_df[consistent_df['overs']>=5.0]
    return(consistent_df)


# Converting the column 'date' from string into datetime object
def coverting_date_time(df):
    print("Before converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    # print("After converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    return(df)



# Get correlation of all the features of the dataset
def plotting_heatmap(df):
    corr_matrix = df.corr()
    top_corr_features = corr_matrix.index
    plt.figure(figsize = (13,10))
    sns.heatmap(data =df[top_corr_features].corr(),annot = True,cmap = 'YlGnBu')
    plt.show()



# Data Preprocessing

def onehot_encoding(df):
    encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
    return(encoded_df)



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



# Linear Regression
def linear_regressor_model(x_train,y_train,x_test):
    from sklearn.linear_model import LinearRegression
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train,y_train)
    # predicting result
    y_pred = linear_regressor.predict(x_test)
    return(y_pred)





# model Evaluation --------> Linear regression 
def model_evaluation(y_test,y_pred):
    from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,accuracy_score
    print("---- Linear Regression - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred))))

# model_evaluation(y_test,y_pred)

# decidion tree
def decision_tree(x_train,y_train,x_test):
    from sklearn.tree import DecisionTreeRegressor
    decision_regressor = DecisionTreeRegressor()
    decision_regressor.fit(x_train,y_train)
    y_pred_dt = decision_regressor.predict(x_test)
    return(y_pred_dt)

# model Evaluation --------> Decision Tree regression 
def model_evaluation_dt(y_test,y_pred_dt):
    from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,accuracy_score
    print("---- Decision Tree Regression - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_dt)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_dt)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_dt))))

# Random Forest Regression
def random_forest_regression(x_train,y_train,x_test):
    from sklearn.ensemble import RandomForestRegressor
    random_regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
    random_regressor.fit(x_train,y_train)
    y_pred_rf = random_regressor.predict(x_test)
    return(y_pred_rf)


# model Evaluation --------> Random Forest regression 
def model_evaluation_rf(y_test,y_pred_rf):
    from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,accuracy_score
    print("---- Random Forest regression  - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_rf)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_rf)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_rf))))

# AdaBoost Model using Linear Regression as the base learner
def adaboost(x_train,y_train,x_test):
    from sklearn.linear_model import LinearRegression
    linear_regressor = LinearRegression()
    from sklearn.ensemble import AdaBoostRegressor
    adb_regressor = AdaBoostRegressor(base_estimator=linear_regressor, n_estimators=100)
    adb_regressor.fit(x_train, y_train)
    # Predicting results
    y_pred_adb = adb_regressor.predict(x_test)
    return(y_pred_adb)

# model Evaluation --------> AdaBoost Model using Linear Regression 
def model_evaluation_adb(y_test,y_pred_adb):
    from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,accuracy_score
    print("---- AdaBoost Model using Linear Regression   - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_adb)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_adb)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_adb))))

## Ridge Regression
def ridge_regression(x_train,y_train,x_test):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    ridge=Ridge()
    parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
    ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
    ridge_regressor.fit(x_train,y_train)
    print(ridge_regressor.best_params_)
    print(ridge_regressor.best_score_)
    y_pred_rdg=ridge_regressor.predict(x_test)
    return(y_pred_rdg)

def model_evaluation_rdg(y_test,y_pred_rdg):
    from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,accuracy_score
    print("\n---- Ridge Regressionn   - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_rdg)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_rdg)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_rdg))))


def plot_graph(y_pred,y_pred_dt,y_pred_rf,y_pred_adb,y_pred_rdg):
    plots = [y_pred,y_pred_dt,y_pred_rf,y_pred_adb,y_pred_rdg]
    titles = ['linear','decision','random_forest','AdaBoost','Ridge regression']
    plt.figure(figsize=(16,9))
    sns.set()
    for i in range(6):
        plt.subplot(2,3,i+1),sns.distplot(plots[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()


if __name__ == "__main__":
    df = remove_unwanted_cols(ipl_df)

    consistent_df = containg_consisting_team(df)

    df= remove_5_overs(consistent_df)

    df = coverting_date_time(df)

    # plotting_heatmap(df)

    df = onehot_encoding(df)

    df = rearranging_cols(df)
    print("after re-arranging df become {}".format(df.head))


    x_train,x_test,y_train,y_test = splitting_train_test(df)
    print("Training set: {} and Test set: {}".format(x_train.shape, x_test.shape))


    y_pred=linear_regressor_model(x_train,y_train,x_test)

    model_evaluation(y_test,y_pred)



    y_pred_dt = decision_tree(x_train,y_train,x_test)

    model_evaluation_dt(y_test,y_pred_dt)



    y_pred_rf = random_forest_regression(x_train,y_train,x_test)

    model_evaluation_rf(y_test,y_pred_rf)

    y_pred_adb=adaboost(x_train,y_train,x_test)

    model_evaluation_adb(y_test,y_pred_adb)

    y_pred_rdg = ridge_regression(x_train,y_train,x_test)  

    model_evaluation_rdg(y_test,y_pred_rdg)
            

    plot_graph(y_pred,y_pred_dt,y_pred_rf,y_pred_adb,y_pred_rdg)




