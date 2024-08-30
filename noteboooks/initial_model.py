# %% [markdown]
# # Forecasting promotion sales 1 week ahead
# 
# ## Explanation of the model
# * The model forecasts the Unit Sales one week ahead in case of a promotion or no-promotion on an article level. 
# * The anonymized parquet dataset used as training data contains daily sales of various AH products in 2016 and 2017
# 
# 
# ## Explanation of the data: 
# * DateKey: date identifier
# * StoreCount: number of stores the article is available at
# * ShelfCapacity: total capacity of shelfs over all stores
# * PromoShelfCapacity: additional ShelfCapacity during promotion 
# * IsPromo: indicator if article is in promotion 
# * ItemNumber: item identification number
# * CategoryCode: catergory identification number (product hierarchy) 
# * GroupCode: group identification number (product hierarchy) 
# * UnitSales: number of consumer units sold

# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import datetime
import math
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option("display.max_columns", 100)

# %% [markdown]
# ### Read data

# %%
path = "data/dataset.csv"

df_prep = pd.read_csv(path, sep=";", header=0)
df_prep.head(5)

# %% [markdown]
# # Data Preparation

# %%
# Log transform the target column!
df_prep['UnitSales'] = np.log(df_prep['UnitSales']) 

# %%
# convert the column DateKey (string) to a date column
df_prep['DateKey'] = pd.to_datetime(df_prep['DateKey'], format='%Y%m%d')
df_prep['month'] = df_prep['DateKey'].dt.month
df_prep['weekday'] = df_prep['DateKey'].dt.weekday

# %%
# Drop null values
df_prep_clean_0 = df_prep[df_prep['UnitSales'].notnull()].copy()
df_prep_clean = df_prep_clean_0[df_prep_clean_0['ShelfCapacity'].notnull()].copy()

# Convert columns to correct format
df_prep_clean['month'] = df_prep_clean['month'].astype('category')
df_prep_clean['weekday'] = df_prep_clean['weekday'].astype('category')

# %%
df_prep.head()

# %% [markdown]
# ## Split data into train and test sets
# Given that we are working with time series data in the sense that there is obvious temporal relations in the data, it is crucial to ensure that when training the model, no information about the future is present.
# 
# We split in a temporal sensible way such that the test set is in the future and could not have been used in training.

# %%
df_to_split = df_prep_clean.copy()

# %%
# We split the data in a train set and a test set, we do this, 80, 20 percent respectively.
nr_of_unique_dates = len(df_to_split.DateKey.unique())
train_split_delta = round(nr_of_unique_dates * 0.8)
train_split_date = df_to_split.DateKey.dt.date.min() + datetime.timedelta(days=train_split_delta)

# %%
def train_test_split(total_df, tr_split_date):
    tr_df = total_df[total_df['DateKey'].dt.date <= tr_split_date].copy()
    tst_df = total_df[total_df['DateKey'].dt.date > tr_split_date].copy()
    return tr_df, tst_df

# %%
train_df, test_df = train_test_split(df_to_split, train_split_date)

# %% [markdown]
# We make categories out of the following columns

# %%
train_df['GroupCode'] = train_df['GroupCode'].astype('category')
train_df['ItemNumber'] = train_df['ItemNumber'].astype('category')
train_df['CategoryCode'] = train_df['CategoryCode'].astype('category')

# %% [markdown]
# ### Filter out items that were not present in the training set

# %%
# determine unique item numbers, and filter the validation and test on these
items_we_train_on = train_df['ItemNumber'].unique()
test_df_filtered = test_df[test_df['ItemNumber'].isin(items_we_train_on)].copy()

# %%
test_df_filtered['GroupCode'] = test_df_filtered['GroupCode'].astype('category')
test_df_filtered['ItemNumber'] = test_df_filtered['ItemNumber'].astype('category')
test_df_filtered['CategoryCode'] = test_df_filtered['CategoryCode'].astype('category')

# %% [markdown]
# At this stage the split has been succesful. We will use the training dataframe to train the model. We use the test dataframe to evaluate the model.

# %%
train_df

# %% [markdown]
# ### Create a dataframe where label and features are included in an appropriate way and add lags

# %%
def add_lagged_feature_to_df(input_df, lag_iterator, feature):
    """
    A function that will expand an input dataframe with lagged variables of a specified feature
    Note that the lag is calculated over time (datekey) but also kept appropriate over itemnumber (article)
    
    input_df: input dataframe that should contain the feature and itemnr
    lag_iterator: an object that can be iterator over, that includes info about the requested nr of lags
    feature: feature that we want to include the lag of in the dataset
    """
    output_df = input_df.copy()
    for lag in lag_iterator:
        df_to_lag = input_df[['DateKey', 'ItemNumber', feature]].copy()
        # we add the nr of days equal to the lag we want
        df_to_lag['DateKey'] = df_to_lag['DateKey'] + datetime.timedelta(days=lag)
        
        # the resulting dataframe contains sales data that is lag days old for the date that is in that row
        df_to_lag = df_to_lag.rename(columns={feature: feature+'_-'+str(lag)})
        
        # we join this dataframe on the original dataframe to add the lagged variable as feature
        output_df = output_df.merge(df_to_lag, how='left', on=['DateKey', 'ItemNumber'])
    # drop na rows that have been caused by these lags
    return output_df

# %%
range_of_lags = [7, 14, 21] # 1 week ago, 2 weeks ago, 3 weeks ago
feature_to_lag = 'UnitSales'

# %%
# make the lags per dataset (no data leakage) and also do the NaN filtering per set
train_df_lag = add_lagged_feature_to_df(train_df, range_of_lags, feature_to_lag)
test_df_lag = add_lagged_feature_to_df(test_df_filtered, range_of_lags, feature_to_lag)

# %%
train_df_lag.head()

# %% [markdown]
# #### Drop Datekey
# This was used for the lag construction, but will not be used in the model

# %%
train_df_lag_clean = train_df_lag.drop(columns=['DateKey'])
test_df_lag_clean = test_df_lag.drop(columns=['DateKey'])

# %%
train_df_lag_clean.info()

# %% [markdown]
# # Modelling

# %%
# We convert the data in the required format for the model (label y and features x)
train_y, train_X = train_df_lag_clean['UnitSales'], train_df_lag_clean.drop(columns=['UnitSales'])
test_y, test_X = test_df_lag_clean['UnitSales'], test_df_lag_clean.drop(columns=['UnitSales'])

# %% [markdown]
# ### Model: [RandomForest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

# %%
# set model settings
rfr = RandomForestRegressor(
    n_estimators=100, max_features=round(len(train_X.columns)/3), max_depth=len(train_X.columns),
)

# %%
# Train the model. Takes a couple of minutes.
rf_model = rfr.fit(train_X, train_y)

# %%
# predictions
rf_y_pred = rf_model.predict(test_X)

# %% [markdown]
# ## Model evaluation

# %%
# calculate RMSE of the log unit sales for the rf model
mean_squared_error(rf_y_pred, test_y, squared=False)

# %%
# and the MAE
mean_absolute_error(rf_y_pred, test_y)

# %% [markdown]
# ## Examples of making a prediction for new inputs

# %%
def convert_log_to_units(prediction):
    return int(math.exp(prediction))

# %%
columns = ['StoreCount', 'ShelfCapacity', 'PromoShelfCapacity', 'IsPromo',
       'ItemNumber', 'CategoryCode', 'GroupCode', 'month', 'weekday',
       'UnitSales_-7', 'UnitSales_-14', 'UnitSales_-21']

# %%
custom_example = pd.DataFrame(
    data=[
        (781, 12602.000, 4922, True, 8646, 7292, 5494, 11, 3, 6.190, 6.217, 6.075),
    ], 
    columns=columns,
)
custom_example_y = 8.187021067343505

# %%
custom_example

# %%
example_pred = rf_model.predict(custom_example)
print(f'Model prediction: {example_pred[0]} which means a predicted UnitSales of {convert_log_to_units(example_pred[0])}')
print(f'Real value is: {custom_example_y} which means a predicted UnitSales of {convert_log_to_units(custom_example_y)}')
print(f'So the delta is {abs(convert_log_to_units(custom_example_y) - convert_log_to_units(example_pred[0]))} units')

# %%
another_example_pred = rf_model.predict([[781, 12602.000, 4922, True, 8646, 7292, 5494, 11, 3, 6.190, 6.217, 6.075]])
convert_log_to_units(another_example_pred)

# %% [markdown]
# ## Saving and loading the model

# %%
# Save the model to disk
filename = 'forecasting_model.pkl'
pickle.dump(rf_model, open(filename, 'wb'))

# %%
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
loaded_from_disk_pred = loaded_model.predict([[781, 12602.000, 4922, True, 8646, 7292, 5494, 11, 3, 6.190, 6.217, 6.075]])
convert_log_to_units(loaded_from_disk_pred)

# %%



