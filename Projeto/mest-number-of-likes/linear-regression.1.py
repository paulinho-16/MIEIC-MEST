#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x)) #Limiting floats output to 5 decimal points

#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##display the first five rows of the train dataset.
print(train.head(5))
print("\n")

##display the first five rows of the test dataset.
print(test.head(5))
print("\n")

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))
#Save the 'Id' column
train_ID = train['ID']
test_ID = test['ID']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("ID", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)


# creating date for analysis
train['date'] = 1000*train['month'] + 100*train['weekday'] + train['hour']
test['date'] = 1000*test['month'] + 100*test['weekday'] + test['hour']

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
print("\n")
  
# fig, ax = plt.subplots(1, 3, figsize=(9, 3))
# ax[0].scatter(x = train['month'], y = train['no_likes'])
# ax[1].bar(x = train['month'], y= train['month'].value_counts())
# plt.xlabel('month', fontsize=13)
# plt.show()

#train.groupby('month').nunique().plot(kind='bar')
#plt.show()

#train.groupby('category').nunique().plot(kind='bar')
#plt.show()

#Visualizing 'outlier' suspects 
print("Suspect outliers:\n", train[(train['hour']>13) | (train['no_likes']>3000) | (train['month']<5)] )
print("\n")
#Deleting outliers
train = train.drop(train[(train['hour']>20) | (train['no_likes']>3000) | (train['month']<5)].index)

#Check the graphic again
# fig, ax = plt.subplots()
# ax.scatter(train['hour'], train['no_likes'])
# plt.ylabel('no_likes', fontsize=13)
# plt.xlabel('hour', fontsize=13)
# plt.show()
print(train['no_likes'].describe())
sns.distplot(train['no_likes'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['no_likes'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('no_likes distribution')

# Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['no_likes'], plot=plt)
plt.show()

############################
#We use the numpy fuction log1p which  applies log(x+1) to all elements of the column
train["no_likes"] = np.log1p(train["no_likes"]) 

#Check the new distribution 
sns.distplot(train['no_likes'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['no_likes'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('no_likes distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['no_likes'], plot=plt)
plt.show()


##################

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.no_likes.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['no_likes'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

#Correlation map to see how features are correlated with no_likes
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()


# As string
all_data['type_of_post'] = all_data['type_of_post'].apply(str)
#Changing category into a categorical variable
all_data['type_of_post'] = all_data['type_of_post'].astype(str)
all_data['category'] = all_data['category'].astype(str)
all_data['month'] = all_data['month'].astype(str)
all_data['weekday'] = all_data['weekday'].astype(str)
all_data['hour'] = all_data['hour'].astype(str)
all_data['paid'] = all_data['paid'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ("type_of_post","category","month","weekday","hour","paid")
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))
skewness = skewness[abs(skewness) > 0.05]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape))

from scipy.special import boxcox1p
skewed_features = ['type_of_post', 'paid', 'category', 'weekday','page_total_likes', 'date']
lam = 0.0005
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = pd.get_dummies(all_data)
print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]

#print(train)
#print(test)

'''''
'''''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

model = LinearRegression().fit(train.values, y_train)
stacked_train_pred = model.predict(train.values)
stacked_pred = np.expm1(model.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
print(rmsle([-1]*150, stacked_pred)) # As RapidMiner


sub = pd.DataFrame()
sub['ID'] = test_ID
sub['no_likes'] = stacked_pred

print(sub.describe())
sub.to_csv('submission2.csv',index=False)