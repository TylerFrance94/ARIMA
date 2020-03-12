import pandas as pd
import numpy as np
import json, requests

pd.set_option("display.max_rows", 1000)


def print_wrapped(data, ncols=3):
    """A helper function to wrap data columns for easier viewing."""
    nrows = len(data)
    labels = data.index
    n_split_rows = int(np.ceil(nrows / ncols))
    for r in range(0, nrows, ncols):
        for c in range(ncols):
            try:
                numstr = '{}'.format(data[r + c])
                tabs = [' '] * (20 - len(labels[r + c]) - len(numstr))
                print(labels[r + c] + "".join(tabs) + numstr, end='\t')
            except:
                pass
        print()


filename = 'raw_iowa_data.csv'
url = 'https://data.iowa.gov/resource/spsw-4jax.json'
options = '?$select=date,county,sum(sale_dollars),sum(sale_liters)&$group=date,county&$limit=10&$offset=1500'
myResponse = requests.get(url + options, verify=True)
print(myResponse.content.decode('utf-8'))


def getCityDailyData(offset=0):
    """
    A function to pull the data from the api
    Args:
        offset (optional, default 0): An offset to the records (if there are more than 50,000)
    Returns:
        A JSON array with the results.
    """

    url = 'https://data.iowa.gov/resource/spsw-4jax.json'
    selectQuery = '?$select=date,county,sum(sale_dollars),sum(sale_liters)&$group=date,county'

    limitQuery = '&$limit=50000'
    if offset > 0:
        offset = '&$offset={}'.format(offset)
        query = url + selectQuery + limitQuery + offset
    else:
        query = url + selectQuery + limitQuery

    # Send the query to the API endpoint
    myResponse = requests.get(query, verify=True)
    jData = ''
    # For successful API call, response code will be 200 (OK)

    try:
        if (myResponse.ok):
            jData = json.loads(myResponse.content.decode('utf-8'))
            print("The response contains {0} properties".format(len(jData)))
        else:
            print(myResponse.status_code)
            print(myResponse.headers)

    except:
        print(myResponse.status_code)
        print(myResponse.headers)

    return jData
# end of get city data function


data_set1 = getCityDailyData()
count = len(data_set1)
offset = count

# this is going to run through as many times as needed to get all the data from the api
while True:
    data_set2 = getCityDailyData(count)
    count = count + len(data_set2)
    ildfs = pd.json_normalize(data_set1)
    ildfs2 = pd.json_normalize(data_set2)
    # creates a CSV when it fishes
    if len(data_set2) < offset:
        df = ildfs.append(ildfs2)
        df.head()
        df.to_csv(filename, index=False)
        print('\n')
        break
