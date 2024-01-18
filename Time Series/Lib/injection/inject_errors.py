from Lib.injection.dirty_accuracy_rows import injection
import pandas as pd
import pickle

path = "../../Datasets/NEweather.csv"
#path = "../../Datasets/PRSA_Data_imputed.csv"
#path = "../../Datasets/ChlorineConcentration.csv"
#path = "../../Datasets/Electrical_Grid.csv"
df = pd.read_csv(path, sep=",")
df_list, outliers_mask = injection(df, seed=8, name='Weather', name_class=['rain'])
#df_list,outliers_mask = injection(df, seed=8, name='PRSA', name_class=['PM2.5','station'])
#df_list, outliers_mask = injection(df, seed=8, name='Chlorine', name_class=['0'])
#df_list, outliers_mask = injection(df, seed=8, name='Electrical', name_class=['stab','stabf'])

df_list[0].to_csv("../../Datasets/NEWeather_injected_outliers_1.csv", sep=",", index=False, na_rep=' NA')
df_list[1].to_csv("../../Datasets/NEWeather_injected_outliers_2.csv", sep=",", index=False, na_rep=' NA')
df_list[2].to_csv("../../Datasets/NEWeather_injected_outliers_3.csv", sep=",", index=False, na_rep=' NA')
df_list[3].to_csv("../../Datasets/NEWeather_injected_outliers_4.csv", sep=",", index=False, na_rep=' NA')
df_list[4].to_csv("../../Datasets/NEWeather_injected_outliers_5.csv", sep=",", index=False, na_rep=' NA')
#df_list[0].to_csv("../../Datasets/PRSA_data_injected_outliers_1.csv", sep=",", index=False, na_rep=' NA')
#df_list[1].to_csv("../../Datasets/PRSA_data_injected_outliers_2.csv", sep=",", index=False, na_rep=' NA')
#df_list[2].to_csv("../../Datasets/PRSA_data_injected_outliers_3.csv", sep=",", index=False, na_rep=' NA')
#df_list[3].to_csv("../../Datasets/PRSA_data_injected_outliers_4.csv", sep=",", index=False, na_rep=' NA')
#df_list[4].to_csv("../../Datasets/PRSA_data_injected_outliers_5.csv", sep=",", index=False, na_rep=' NA')
#df_list[0].to_csv("../../Datasets/Chlorine_injected_outliers_1.csv", sep=",", index=False, na_rep=' NA')
#df_list[1].to_csv("../../Datasets/Chlorine_injected_outliers_2.csv", sep=",", index=False, na_rep=' NA')
#df_list[2].to_csv("../../Datasets/Chlorine_injected_outliers_3.csv", sep=",", index=False, na_rep=' NA')
#df_list[3].to_csv("../../Datasets/Chlorine_injected_outliers_4.csv", sep=",", index=False, na_rep=' NA')
#df_list[4].to_csv("../../Datasets/Chlorine_injected_outliers_5.csv", sep=",", index=False, na_rep=' NA')
#df_list[0].to_csv("../../Datasets/Electrical_injected_outliers_1.csv", sep=",", index=False, na_rep=' NA')
#df_list[1].to_csv("../../Datasets/Electrical_injected_outliers_2.csv", sep=",", index=False, na_rep=' NA')
#df_list[2].to_csv("../../Datasets/Electrical_injected_outliers_3.csv", sep=",", index=False, na_rep=' NA')
#df_list[3].to_csv("../../Datasets/Electrical_injected_outliers_4.csv", sep=",", index=False, na_rep=' NA')
#df_list[4].to_csv("../../Datasets/Electrical_injected_outliers_5.csv", sep=",", index=False, na_rep=' NA')


with open('../../Datasets/outliers_index_1.pkl', 'wb') as pick:
    pickle.dump(outliers_mask[0],pick)

with open('../../Datasets/outliers_index_2.pkl', 'wb') as pick:
    pickle.dump(outliers_mask[1],pick)

with open('../../Datasets/outliers_index_3.pkl', 'wb') as pick:
    pickle.dump(outliers_mask[2],pick)

with open('../../Datasets/outliers_index_4.pkl', 'wb') as pick:
    pickle.dump(outliers_mask[3], pick)

with open('../../Datasets/outliers_index_5.pkl', 'wb') as pick:
    pickle.dump(outliers_mask[4], pick)

