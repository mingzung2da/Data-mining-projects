import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Read in outcomes data
data_folder = "data/"
outcomes = pd.read_csv(data_folder + 'seda_school_pool_cs_4.1.csv')
# keep only the columns sedasch and cs_mn_avg_ol
outcomes = outcomes[['sedasch','cs_mn_avg_ol']]
# show head
outcomes.head()

# Read in covariates data
covariates = pd.read_csv(data_folder + 'seda_cov_school_pool_4.1.csv')
# keep only the columns we need
covariates = covariates[['sedasch','stateabb','type','level','charter','magnet','urbanicity','locale','totenrl','perwht','pernam','perasn','perhsp','perblk','perfl','perrl','gifted_flag','lep_flag','sped_flag','avgrdall']]
# show head
covariates.head()
# Read in characteristics data
characteristics = pd.read_csv(data_folder + 'Public_School_Characteristics_2017-18.csv')
# keep only the columns we need
characteristics = characteristics[['NCESSCH','GSLO','GSHI','VIRTUAL','TOTAL','STUTERATIO','STITLEI','TOTMENROL','TOTFENROL']]
# show head
characteristics.head()
# Read in poverty data
poverty = pd.read_csv(data_folder + 'Poverty_Data_2017-18.csv')
# keep only the columns we need
poverty = poverty[['NCESSCH','IPR_EST']]
# show head
poverty.head()
# change all the colimn names from upper case to lower case
characteristics.columns = characteristics.columns.str.lower()
poverty.columns = poverty.columns.str.lower()

# rename columns of school id
characteristics = characteristics.rename(columns={'ncessch':'sedasch'})
poverty = poverty.rename(columns={'ncessch':'sedasch'})
# combine via inner join the tables outcomes, covariates, characteristics, and poverty using sedasch as the key(to match the rows)
df = outcomes.merge(covariates, on='sedasch', how='inner').merge(characteristics, on='sedasch', how='inner').merge(poverty, on='sedasch', how='inner')
print(df.shape)
df.head()

# how='inner': keeps only schools in all 4 tables. Drops null values automatically

categorical_variables = ['stateabb','type','level','charter','magnet','urbanicity','locale','gifted_flag','lep_flag','sped_flag','gslo','gshi','virtual','stitlei'] # 14
numerical_variables = ['cs_mn_avg_ol','totenrl','perwht','pernam','perasn','perhsp','perblk','perfl','perrl','avgrdall','total','stuteratio','totmenrol','totfenrol','ipr_est'] # 15

print("categorical variables\n")
for col in categorical_variables:
    print(col)
    print(df[col].unique())
    print(df[col].value_counts())
    print('\n')

print("numerical variables\n")
for col in numerical_variables:
    print(col)
    print(df[col].describe())
    print('\n')

# turn empty or 'Missing' values into nan (.nan=missing/null value)
df = df.replace('', np.nan)
df = df.replace('Missing', np.nan)

# count the number of nan values in each column
for col in df.columns:
    print(col)
    print(df[col].isna().sum())
    print('\n')
# Create a separate dataframe using only the rows of df with missing cs_mn_avg_ol values
df_missing = df[df['cs_mn_avg_ol'].isna()]

for col in categorical_variables:
    print(col)
    print(df_missing[col].unique())
    print(df_missing[col].value_counts()/df[col].value_counts())
    # print(df_missing[col].value_counts()/len(df_missing[col]))
    print('\n')

#why use categorical variable? -> We use the missing values in cs_mn_avg_ol as a filter,
# then examine the categorical variables to understand what types of schools have missing achievement data 

# Question: why there were 0 missing values about the cs_mn_avg_ol value in the pior result, but here we found lots of missing values? 
# remove rows with any nan
df = df.dropna()
print(df.shape)
df
# Full dataframe view
pd.set_option('display.max_rows', None)  # Remove row limit
pd.set_option('display.max_columns', None)  # Remove column limit
print(df)
#quick check before we work on our main dataset
# Create a new dataframe called df_N where either gslo or gshi are 'N ' (note the space after the N)
df_N = df[(df['gslo'] == 'N ') | (df['gshi'] == 'N ')]

for col in categorical_variables:
    print(col)
    print(df_N[col].unique())
    print(df_N[col].value_counts())
    print('\n')

for col in numerical_variables:
    print(col)
    print(df_N[col].describe())
    print('\n')

# To me, it is not obvious, but there are only 2 of them. Let's remove them.
df = df[df['gslo'] != 'N ']
# Mini task 1
# Make new columns called permale and perfemale which contain the fraction of students listed as male or female (using the 2017-2018 data)
df['permale'] = df['totmenrol'] / df['total']
df['perfemale'] = df['totfenrol'] / df['total']

# Mini task 2
# Make new columns called perurm and pernonurm which contain the fraction of students who are URM (Under Represented Minority) including Native Americans, Hispanic, and African Americans, and who are not URM including Caucasians and Asians
df['perurm'] = (df['pernam'] + df['perhsp'] + df['perblk'])
df['pernonurm'] = (df['perwht'] + df['perasn'])

df.head()
# Filter the dataframe so that type has only 'Regular School', level has only 'Middle' 'Elementary' 'High', there are no charter or magnet schools, and no virtual schools. Call the new dataframe df_filtered.
df_filtered = df[(df['type'] == 'Regular School') & (df['level'].isin(['Middle','Elementary','High'])) & (df['charter'] == 0) & (df['magnet'] == 0) & (df['virtual'] == 'Not a virtual school')]

for col in categorical_variables:
    print(col)
    print(df_filtered[col].unique())
    print(df_filtered[col].value_counts())
    print('\n')

for col in numerical_variables:
    print(col)
    print(df_filtered[col].describe())
    print('\n')
# remove row where a numerical variables if outside the [0.5, 99.5] percentile range
for col in numerical_variables:
    indlower = df_filtered[col] > df_filtered[col].quantile(0.005)
    indupper = df_filtered[col] < df_filtered[col].quantile(0.995)
    ind = indlower & indupper
    df_filtered = df_filtered[ind]   
# size of resulting dataframe
df_filtered.shape
# save dataframe to csv
fname = data_folder + 'analyzed_seda_plus.csv'
df_filtered.to_csv(fname, index=False)

# check if the file has succesfully saved in your folder