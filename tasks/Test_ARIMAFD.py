import pandas as pd
import numpy as np
import arimafd as oa

my_array = np.random.normal(size=1000) # init array
my_array[-3] = 1000 # init anomaly
ts = pd.DataFrame(my_array,
                  index=pd.date_range(start='01-01-2000',
                                      periods=1000,
                                      freq='H'))

my_arima = oa.Arima_anomaly_detection()
my_arima.fit(ts[:500])
ts_anomaly = my_arima.predict(ts[500:])


# or you can use for streaming:
# bin_metric = []
# for i in range(len(df)):
#     bin_metric.append(my_arima.predict(df[i:i+1]))
# bin_metric = pd.concat(bin_metric)
# bin_metric
print(ts_anomaly)