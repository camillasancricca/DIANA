import numpy as np
import pandas as pd
from Lib import preproc_lib as pp


def max_na(s):
    isna = s.isna()
    blocks = (~isna).cumsum()
    return isna.groupby(blocks).sum().max()


df_list = []

#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Aotizhongxin_20130301-20170228.csv"))  #NO
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Changping_20130301-20170228.csv")) #Rimuovere CO? Non bellissimo
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Dingling_20130301-20170228.csv")) #NO
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Dongsi_20130301-20170228.csv")) #NO
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Guanyuan_20130301-20170228.csv")) #NO
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Gucheng_20130301-20170228.csv")) #NO
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Huairou_20130301-20170228.csv")) #BUCHI
df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Nongzhanguan_20130301-20170228.csv")) #DROP CO
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Shunyi_20130301-20170228.csv")) #NO
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Tiantan_20130301-20170228.csv")) #DROP SO2
#df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Wanliu_20130301-20170228.csv")) #NO
df_list.append(pd.read_csv("../Datasets/Beijing Multi-Site Air-Quality Data Set/PRSA_Data_Wanshouxigong_20130301-20170228.csv")) #INSOMMA

for i in range(len(df_list)):
    df_list[i]['station'] = i
    df_list[i]['date_string'] = df_list[i]['year'].astype(str) + '-' + df_list[i]['month'].astype(str).str.zfill(2) + '-' + df_list[i]['day'].astype(str).str.zfill(2) + ' ' + df_list[i]['hour'].astype(str).astype(str).str.zfill(2)

    df_list[i]['date_time'] = pd.to_datetime(df_list[i]['date_string'], format='%Y-%m-%d %H')

    df_list[i].drop(['date_string', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)

    desired_order = df_list[i].columns.tolist()
    desired_order.remove('date_time')
    desired_order.insert(1, 'date_time')
    df_list[i] = df_list[i].reindex(columns=desired_order)

    encoding_map = {'N': 1, 'NNE': 2, 'NE': 3, 'ENE': 4, 'E': 5, 'ESE': 6, 'SE': 7, 'SSE': 8, 'S': 9, 'SSW': 10,
                    'SW': 11,
                    'WSW': 12, 'W': 13, 'WNW': 14, 'NW': 15, 'NNW': 16}

    df_list[i]['wd'] = df_list[i]['wd'].replace(encoding_map)
    df_list[i]['wd'] = df_list[i]['wd'].replace('', np.nan, regex=True)

merged_df = pd.concat(df_list).sort_values(by=['date_time', 'station']).reset_index(drop=True)

merged_df.drop('No', axis=1, inplace=True)
merged_df['station'] = merged_df['station'].astype(int)

print(merged_df)
# merged_df = merged_df.drop(['O3','CO'],axis=1)
#sequence = np.array(merged_df.dropna(how='any').index)
#longest_seq = max(np.split(sequence, np.where(np.diff(sequence) != 1)[0] + 1), key=len)
#print(merged_df.iloc[longest_seq])


#print(merged_df.isna().sum())

#print(merged_df.apply(max_na).max())
print(len(merged_df.dropna())/len(merged_df))


merged_df.to_csv("../Datasets/PRSA_Data_complete.csv", na_rep=' NA', index=False)
merged_df.dropna().to_csv("../Datasets/PRSA_Data_NoMiss.csv", na_rep=' NA', index=False)