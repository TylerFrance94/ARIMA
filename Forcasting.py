import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import matplotlib.gridspec as gridspec
from pmdarima.arima import auto_arima
from DataExploration import state_data, df
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose


state_data_train = state_data[state_data.index <= '2017-06-01']
state_data_test = state_data[state_data.index > '2017-06-01']

# Get the dates from the test data
test_months = state_data_test.reset_index(drop=False)['month']

# Offset them by a year
previous_year = test_months.apply(lambda x: x - relativedelta(years=1))

# Get the values from the previous year
last_year_preds = state_data.loc[previous_year, 'sum_sale_liters'].values


fig_1, ax = plt.subplots( figsize=(15,5))
clrs = sns.color_palette("husl", 5)

# #############################################################################
# Plot actual test vs. forecasts:
plt.plot(state_data_train.index, state_data_train['sum_sale_liters'], marker='o', c=clrs[0])

plt.plot(state_data_test.index, state_data_test['sum_sale_liters'], marker='o', c=clrs[1])

plt.plot(state_data_test.index, last_year_preds, linestyle='None', marker='x', c=clrs[2])

plt.title("Actual test samples vs. forecasts -- Last Year's Number")
plt.ylabel('State Total Sales [liters]')
plt.tight_layout()
plt.savefig("ActualTestSamplesVsForecastsLastYearsNumber.png")
# plt.show()
plt.close()
fig_1


# Fit a simple auto_arima model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    state_arima = auto_arima(state_data_train['sum_sale_liters'],
                                 error_action='ignore',
                                 trace=True,
                                 seasonal=True,
                                 m=12)

state_arima.summary()

predictions = state_arima.predict(n_periods=state_data_test.shape[0], return_conf_int=True)
pred_mean = predictions[0]
pred_interval = predictions[1]

fig_1, ax = plt.subplots( figsize=(15,5))
clrs = sns.color_palette("husl", 5)

# #############################################################################
# Plot actual test vs. forecasts:
plt.plot(state_data_train.index, state_data_train['sum_sale_liters'], marker='o', c=clrs[0])

plt.plot(state_data_test.index, state_data_test['sum_sale_liters'], marker='o', c=clrs[1])

plt.plot(state_data_test.index, last_year_preds, linestyle='None', marker='x', c=clrs[2])

plt.plot(state_data_test.index, pred_mean, linestyle='None', marker='_', c=clrs[3])
plt.fill_between(state_data_test.index, pred_interval[:,0], pred_interval[:,1], alpha=0.2, facecolor=clrs[3])

plt.title('Actual test samples vs. forecasts -- 18 month forecast')
plt.ylabel('State Total Sales [liters]')
plt.tight_layout()
plt.savefig('ActualTestSamplesVsForecasts18MonthForecast.png')
# plt.show()
plt.close()
fig_1

fig_1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
ax1.bar(range(len(state_data_test.index)), (pred_interval[:,1] - pred_mean))
ax1.set_xticklabels(state_data_test.index.strftime('%Y-%m'), rotation=40)
ax1.set_ylabel("Forecast Confidence Interval [liters]")

ax2.bar(range(len(state_data_test.index)), (pred_interval[:,1] - pred_mean)/(pred_mean)*100)
ax2.set_xticklabels(state_data_test.index.strftime('%Y-%m'), rotation=40)
ax2.set_ylabel("Forecast Relative Confidence Interval (%)")
plt.savefig("ForecastRelativeConfidenceInterval.png")
# plt.show()

start_date = datetime.datetime(2017, 6, 1)

r = relativedelta(state_data.index.max(), start_date)
n_months = r.months + r.years * 12 - 1

prediction_df = pd.DataFrame({'date': [], 'actual': [], 'pred': [], 'pred_interval_low': [], 'pred_interval_high': []})

for months_from_start in range(n_months):
    current_split_month = (start_date + relativedelta(months=months_from_start)).strftime('%Y-%m-%d')

    state_data_train = state_data[state_data.index <= current_split_month]
    state_data_test = state_data[state_data.index > current_split_month]
    print("Splitting at {} with {} months in train and {} in test".format(current_split_month,
                                                                          len(state_data_train),
                                                                          len(state_data_test)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state_arima = auto_arima(state_data_train['sum_sale_liters'], error_action='ignore', trace=False,
                                 seasonal=True, m=12)

    predict_month = start_date + relativedelta(months=(months_from_start + 2))
    predictions = state_arima.predict(n_periods=2, return_conf_int=True)
    pred_mean = predictions[0][-1]
    pred_interval = predictions[1][-1]

    actual = state_data_test.loc[predict_month.strftime('%Y-%m-%d'), 'sum_sale_liters']

    pred_interval_low = pred_interval[0]
    pred_interval_high = pred_interval[1]
    prediction_df = prediction_df.append({'date': predict_month,
                                          'actual': actual,
                                          'pred': pred_mean,
                                          'pred_interval_low': pred_interval_low,
                                          'pred_interval_high': pred_interval_high}, ignore_index=True)

prediction_df.set_index('date', inplace=True)


fig_2, ax = plt.subplots( figsize=(15,5))
clrs = sns.color_palette("husl", 5)

# #############################################################################
# Plot actual test vs. forecasts:
plt.plot(state_data.index, state_data['sum_sale_liters'], c=clrs[0])

plt.plot(prediction_df.index, prediction_df['actual'], marker='o', c=clrs[1])

plt.plot(prediction_df.index, prediction_df['pred'], marker='_', linestyle='None', c=clrs[3])
plt.fill_between(prediction_df.index, prediction_df['pred_interval_low'], prediction_df['pred_interval_high'], alpha=0.2, facecolor=clrs[3])

plt.title('Actual test samples vs. forecasts -- rolling two-month forecast')
plt.ylabel('Sales [liters]')
plt.tight_layout()
plt.savefig('ActualTestSamplesVsForecastsRollingTwoMonthForecast.png')
# plt.show()
plt.close()
fig_2

fig_1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))
ax1.bar(range(len(prediction_df.index)),
       (prediction_df['pred_interval_high'] - prediction_df['pred']))
ax1.set_xticklabels(prediction_df.index.strftime('%Y-%m'), rotation=40)
ax1.set_ylabel("Forecast Confidence Interval [±liters]")

ax2.bar(range(len(prediction_df.index)),
        (prediction_df['pred_interval_high'] - prediction_df['pred'])/(prediction_df['pred'])*100)
ax2.set_xticklabels(prediction_df.index.strftime('%Y-%m'), rotation=40)
ax2.set_ylabel("Forecast Relative Confidence Interval (%)")
plt.suptitle("Rolling 2-month-out Forecast")
# plt.show()
plt.savefig("Rolling2MonthOutForecast.png")

county_data = df.groupby(['month', 'county']).sum().reset_index(drop=False, level=1)

fig, ax = plt.subplots(figsize=[15, 8])

for county in county_data['county'].unique():
    county_data.loc[county_data['county'] == county, 'sum_sale_liters'].plot(ax=ax, label=county)

ax.set_yscale("log", nonposy='clip')
ax.set_ylabel("County Sales [liters]")
# plt.show()
plt.savefig("CountrySalesLiters.png")

df.groupby('county').sum().sort_values('sum_sale_liters', ascending=False).head()


df_county = df[df['county'] == 'POLK'].groupby('month').sum()
df_county['sum_sale_liters'].plot(figsize=[15,5])
plt.ylabel("POLK County Sales [liters]")

res = seasonal_decompose(df_county['sum_sale_liters'], model='additive', two_sided=False)
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,8))
res.trend.plot(ax=ax1)
ax1.set_ylabel('Trend')
res.seasonal.plot(ax=ax2)
ax2.set_ylabel('Seasonal')
res.resid.plot(ax=ax3)
ax3.set_ylabel('Residuals')
plt.tight_layout()

df_county = df[df['county'] == 'DICKINSON'].groupby('month').sum()
df_county['sum_sale_liters'].plot(figsize=[15,5])
plt.ylabel("DICKINSON County Sales [liters]")

res = seasonal_decompose(df_county['sum_sale_liters'], model='additive', two_sided=False)
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,8))
res.trend.plot(ax=ax1)
ax1.set_ylabel('Trend')
res.seasonal.plot(ax=ax2)
ax2.set_ylabel('Seasonal')
res.resid.plot(ax=ax3)
ax3.set_ylabel('Residuals')
plt.tight_layout()


def forecast_county(df_county):
    """A function to create the rolling 2-month-out forecast"""

    start_date = datetime.datetime(2017, 6, 1)

    r = relativedelta(df_county.index.max(), start_date)
    n_months = r.months + r.years * 12 - 1

    prediction_county = pd.DataFrame(
        {'date': [], 'actual': [], 'pred': [], 'pred_interval_low': [], 'pred_interval_high': []})

    for months_from_start in range(n_months):
        current_split_month = (start_date + relativedelta(months=months_from_start)).strftime('%Y-%m-%d')

        df_county_train = df_county[df_county.index <= current_split_month]
        df_county_test = df_county[df_county.index > current_split_month]
        print("Splitting at {} with {} months in train and {} in test".format(current_split_month,
                                                                              len(df_county_train),
                                                                              len(df_county_test)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dickinson_arima = auto_arima(df_county_train['sum_sale_liters'], error_action='ignore', trace=False,
                                         seasonal=True, m=12)

        predict_month = start_date + relativedelta(months=(months_from_start + 2))
        predictions = dickinson_arima.predict(n_periods=2, return_conf_int=True)
        pred_mean = predictions[0][-1]
        pred_interval = predictions[1][-1]

        actual = df_county_test.loc[predict_month.strftime('%Y-%m-%d'), 'sum_sale_liters']

        pred_interval_low = pred_interval[0]
        pred_interval_high = pred_interval[1]
        prediction_county = prediction_county.append({'date': predict_month,
                                                      'actual': actual,
                                                      'pred': pred_mean,
                                                      'pred_interval_low': pred_interval_low,
                                                      'pred_interval_high': pred_interval_high}, ignore_index=True)

    prediction_county.set_index('date', inplace=True)
    return prediction_county


def show_plots(df_county, prediction_county, county_name=""):
    """A function to plot the rolling 2-month-out forecast results"""

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])

    clrs = sns.color_palette("husl", 5)

    # #############################################################################
    # Plot actual test vs. forecasts:
    ax1.plot(df_county.index, df_county['sum_sale_liters'], c=clrs[0])

    ax1.plot(prediction_county.index, prediction_county['actual'], marker='o', c=clrs[1])

    ax1.plot(prediction_county.index, prediction_county['pred'], marker='_', linestyle='None', c=clrs[3])
    ax1.fill_between(prediction_county.index, prediction_county['pred_interval_low'],
                     prediction_county['pred_interval_high'], alpha=0.2, facecolor=clrs[3])

    ax1.set_ylabel('{} County Sales [liters]'.format(county_name))

    ax2.bar(range(len(prediction_county.index)),
            (prediction_county['pred_interval_high'] - prediction_county['pred']))
    ax2.set_xticklabels(prediction_county.index.strftime('%Y-%m'), rotation=40)
    ax2.set_ylabel("Forecast Confidence Interval [±liters]")

    ax3.bar(range(len(prediction_county.index)),
            (prediction_county['pred_interval_high'] - prediction_county['pred']) / (prediction_county['pred']) * 100)
    ax3.set_xticklabels(prediction_df.index.strftime('%Y-%m'), rotation=40)
    ax3.set_ylabel("Forecast Relative Confidence Interval (%)")
    plt.suptitle("Rolling 2-month-out Forecast for {} County".format(county_name))
    plt.savefig("Rolling2MonthOutForcastForCounty.png")
    # plt.show()


prediction_county = forecast_county(df_county)
show_plots(df_county, prediction_county, 'DICKINSON')
