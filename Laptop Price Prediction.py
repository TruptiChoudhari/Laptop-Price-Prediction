#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Load the Data

# In[2]:


df=pd.read_csv("E:\Cuvette\Python & ML\Mini Project\laptop_price - dataset.csv")


# ### Data Inspection 

# In[3]:


df


# In[4]:


df.head()


# In[5]:


df["OpSys"].nunique()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.size


# In[10]:


df.describe()


# ### Checking Null Values

# In[11]:


df.isnull().sum()


# -  There are no missing values

# In[12]:


print(df.duplicated().sum())


# -  No duplicate values

# ### Checking outliers 

# In[13]:


sns.boxplot(data=df, x="Inches")


# In[14]:


Q1 = df['Inches'].quantile(0.25)
Q3 = df['Inches'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print outliers
outliers = df[(df['Inches'] < lower_bound) | (df['Inches'] > upper_bound)]
print(outliers)


# -  Here, Inches column is representing screen size of laptop, which typically ranges from 11" to 17". And after seeing the outliers it can be concluded that these values are valid, considering the high end laptop and small sized laptop. 

# In[15]:


sns.boxplot(data=df, x="CPU_Frequency (GHz)")


# In[16]:


Q1 = df['CPU_Frequency (GHz)'].quantile(0.25)
Q3 = df['CPU_Frequency (GHz)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print outliers
outliers = df[(df['CPU_Frequency (GHz)'] < lower_bound) | (df['CPU_Frequency (GHz)'] > upper_bound)]
print(outliers)


# In[17]:


#We will drop the outlier for this category
df = df[df['CPU_Frequency (GHz)'] > 0.9]


# In[18]:


sns.boxplot(data=df, x="CPU_Frequency (GHz)")


# -  No more outliers in  CPU_Frequency(GHz)

# In[19]:


sns.boxplot(data=df, x="RAM (GB)")


# In[20]:


Q1 = df['RAM (GB)'].quantile(0.25)
Q3 = df['RAM (GB)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print outliers
outliers = df[(df['RAM (GB)'] < lower_bound) | (df['RAM (GB)'] > upper_bound)]
print(outliers)


# -  From above data it can interpreted that data as outlier is valid values

# In[21]:


sns.boxplot(data=df, x="Weight (kg)")


# In[22]:


Q1 = df['Weight (kg)'].quantile(0.25)
Q3 = df['Weight (kg)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print outliers
outliers = df[(df['Weight (kg)'] < lower_bound) | (df['Weight (kg)'] > upper_bound)]
print(outliers)


# ### Exploratory data Analysis

# In[23]:


df.info()


# ##### Univariate Analysis

# In[24]:


#Numerical Values
numerical_columns=["Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)", "Price (Euro)"]

#Histogram
for col in numerical_columns:
    plt.figure(figsize=(10,5))
    sns.histplot(data=df, x=col)
    plt.show
    
#Boxplot
for col in numerical_columns:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x=col)
    plt.show   


# In[25]:


#Categorical Values
categorical_columns=["Company","TypeName","CPU_Company","GPU_Company","OpSys"]
#Histogram
for col in categorical_columns:
    plt.figure(figsize=(20,7))
    sns.histplot(data=df, x=col, palette='coolwarm')
    plt.show


# ### Bivariate Analysis

# #### Relationship between Categorical columns and Price(target)

# In[26]:


categorical_columns=[("Company","Price (Euro)"),("TypeName","Price (Euro)"),("CPU_Company","Price (Euro)"),("GPU_Company","Price (Euro)"),("OpSys","Price (Euro)")]
for c1, c2 in categorical_columns:
    plt.figure(figsize=(25,10))
    sns.boxplot(data=df, x=c1, y=c2)


# ##### Relationship between numerical columns and Price(Target)

# In[27]:


numerical_columns=[("Inches","Price (Euro)"), ("CPU_Frequency (GHz)","Price (Euro)"), ("RAM (GB)","Price (Euro)"), ("Weight (kg)","Price (Euro)")]
for c1, c2 in numerical_columns:
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=df, x=c1, y=c2)
    


# #### Heatmap

# In[28]:


plt.figure(figsize=(10,5))
cor_matrix=df.corr()
sns.heatmap(cor_matrix,annot=True)
plt.title("Correlation Matrix Heatmap")


# #### Encoding 

# In[29]:


def simplify_resolution(res):
    if '1366x768' in res:
        return 'HD'
    elif '1920x1080' in res:
        return 'Full HD'
    elif '2560x1440' in res or '2560x1600' in res or '3200x1800' in res:
        return 'Quad HD+'
    elif '3840x2160' in res:
        return '4K Ultra HD'
    elif 'Retina' in res or '2304x1440' in res or '2736x1824' in res or '2880x1800' in res:
        return 'Retina'
    else:
        return 'Other'

# Apply the function
df['SimplifiedResolution'] = df['ScreenResolution'].apply(simplify_resolution)

print(df['SimplifiedResolution'].value_counts())

#df["ScreenResolution"].unique()


# In[30]:


def simplify_gpu_type(gpu):
    if 'GeForce' in gpu or 'GTX' in gpu or 'MX' in gpu:
        return 'NVIDIA GeForce'
    elif 'Quadro' in gpu:
        return 'NVIDIA Quadro'
    elif 'Radeon' in gpu or 'FirePro' in gpu:
        return 'AMD Radeon'
    elif 'Iris' in gpu or 'HD Graphics' in gpu:
        return 'Intel Integrated'
    elif 'Mali' in gpu:
        return 'ARM Mali'
    else:
        return 'Other'

# Apply the function
df['SimplifiedGPU'] = df['GPU_Type'].apply(simplify_gpu_type)
print(df['SimplifiedGPU'].value_counts())


# In[32]:


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pandas as pd

label_encoder = LabelEncoder()

# Product Label Encoding
df['Company_Encoded'] = label_encoder.fit_transform(df['Company'])

# One-hot encoding for 'TypeName'
df = pd.get_dummies(df, columns=['TypeName'], prefix='TypeName')

# Screen Resolution Encoding (Ordinal)
resolution_order = ['HD', 'Full HD', 'Quad HD+', '4K Ultra HD', 'Retina', 'Other']
ordinal_encoder = OrdinalEncoder(categories=[resolution_order])
df['ScreenResolution_Encoded'] = ordinal_encoder.fit_transform(df[['SimplifiedResolution']])

# CPU_Company One-hot Encoding
df = pd.get_dummies(df, columns=['CPU_Company'], prefix='CPU_Company', drop_first=True)

# CPU_Type Label Encoding
df['CPU_Type_Encoded'] = label_encoder.fit_transform(df['CPU_Type'])

# Function to extract HDD and SSD values
def extract_storage(memory_col, storage_type):
    storage = memory_col.str.extract(f'(\d+)(?:GB|TB) {storage_type}')
    return pd.to_numeric(storage[0], errors='coerce').fillna(0).apply(lambda x: x * 1024 if 'TB' in memory_col else x)

# Create separate columns for SSD and HDD
df['SSD (GB)'] = extract_storage(df['Memory'], 'SSD')
df['HDD (GB)'] = extract_storage(df['Memory'], 'HDD')

# GPU_Company One-hot Encoding
df = pd.get_dummies(df, columns=['GPU_Company'], prefix='GPU_Company', drop_first=True)

# GPU_Type Encoding (Nominal)
df = pd.get_dummies(df, columns=['SimplifiedGPU'], prefix='SimplifiedGPU', drop_first=True)

# OpSys One-hot Encoding
df = pd.get_dummies(df, columns=['OpSys'], prefix='OpSys', drop_first=True)


# In[33]:


df.info()


# In[34]:


df.drop(columns=['Company','Product','ScreenResolution','CPU_Type', 'Memory','GPU_Type', 'SimplifiedResolution'], inplace=True)


# In[35]:


df


# ### Model Building

# In[36]:


x=df.drop("Price (Euro)",axis=1)
y=df[["Price (Euro)"]]


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=80)


# In[39]:


x_train.head()


# In[40]:


x_test.head()


# In[41]:


y_train.head()


# In[42]:


y_test.head()


# ### Linear Regression

# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[44]:


# Initialize and train the Linear Regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train)
lr_predictions = linear_regression_model.predict(x_test)
print("Linear Regression R-squared:", r2_score(y_test, lr_predictions))
print("Linear Regression MAE:", mean_absolute_error(y_test, lr_predictions))
print("Linear Regression MSE:", mean_squared_error(y_test, lr_predictions))


# ### Decision Tree 

# In[45]:


from sklearn.tree import DecisionTreeRegressor


# In[46]:


decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(x_train, y_train)
dt_predictions = decision_tree_model.predict(x_test)
print("Decision Tree R-squared:", r2_score(y_test, dt_predictions))
print("Decision Tree MAE:", mean_absolute_error(y_test, dt_predictions))
print("Decision Tree MSE:", mean_squared_error(y_test, dt_predictions))


# ### Random Forest

# In[47]:


from sklearn.ensemble import RandomForestRegressor


# In[48]:


random_forest_model = RandomForestRegressor()
random_forest_model.fit(x_train, y_train)
rf_predictions = random_forest_model.predict(x_test)
print("Random Forest R-squared:", r2_score(y_test, rf_predictions))
print("Random Forest MAE:", mean_absolute_error(y_test, rf_predictions))
print("Random Forest MSE:", mean_squared_error(y_test, rf_predictions))


# ### Observation
# -  Linear Regression Model has R² Score = 72%
# -  Decision Trre Model has R² Score = 65%
# -  Random Forest Model has R² Score = 75%
# 
# -- This indicates that Random Forest Model will be best evaluation model. 

# In[ ]:




