import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from MakeCSV import filename, print_wrapped


df = pd.read_csv(filename)
df.dtypes

df['date'] = pd.to_datetime(df['date'])
print(df.dtypes)
df.head()

# let's trim off the current month since we only want full month data
current_month = datetime.datetime.now().strftime('%Y-%m-01')
print("Initial last date: {}".format(df['date'].max()))

# Filter rows using the date column
df = df[df['date'] < current_month]
print("After filter last date: {}".format(df['date'].max()))
print('\n')

# this is going to print out what counties we have and the
county_rows = df.groupby('county').date.count()
print_wrapped(county_rows, ncols=4)
print('\n')

# let's sort by the county name
df['county'] = df['county'].str.upper()
df.dropna(inplace=True)

# we're just renaming a few counties to their actual names
df.loc[df['county'] == 'BUENA VIST','county'] = 'BUENA VISTA'
df.loc[df['county'] == 'CERRO GORD','county'] = 'CERRO GORDO'
df.loc[df['county'] == 'OBRIEN','county'] = "O'BRIEN"
df.loc[df['county'] == 'POTTAWATTA','county'] = "POTTAWATTAMIE"

# let's get El Paso out of here
df = df.loc[df['county'] != 'EL PASO']

# once again resorting the data by the counties post name change
col_v2 = df.groupby('county').date.count()
print_wrapped(col_v2, ncols=4)

# let's create a new column labeled month and aggregate the sales based on that column
df['month'] = df['date'].apply(lambda x: x.strftime('%Y-%m-01'))
df['month'] = pd.to_datetime(df['month'])
df.head()

# plotting the data, so we can get a visualization
state_data = df.groupby('month').sum()
state_data['sum_sale_liters'].plot(figsize=[15,5])
plt.ylabel('State-wide Sales [liters]')
# plt.show()


x = np.arange('2012-01', '2019-02', dtype='datetime64[M]')
xm = (x - np.min(x)).astype('int')
y = 2.3 + 0.2 * xm + np.cos(2 * np.pi * xm / 12)
fig, ax = plt.subplots(figsize=[15,5])
ax.plot(x,y)
ax.set_ylabel("Test Data")
# plt.show()

xy = pd.DataFrame(y, x, columns=['y'])

res = seasonal_decompose(xy, model='additive', two_sided=False)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5))
res.trend.plot(ax=ax1)
ax1.set_ylabel('Trend')
res.seasonal.plot(ax=ax2)
ax2.set_ylabel('Seasonal')
res.resid.plot(ax=ax3)
ax3.set_ylabel('Residuals')
plt.tight_layout()
# plt.show()

res = seasonal_decompose(state_data['sum_sale_liters'], model='additive', two_sided=False)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
res.trend.plot(ax=ax1)
ax1.set_ylabel('Trend')
res.seasonal.plot(ax=ax2)
ax2.set_ylabel('Seasonal')
res.resid.plot(ax=ax3)
ax3.set_ylabel('Residuals')
plt.tight_layout()
# plt.show()
print('\n')
