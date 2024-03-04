import pandas as pd
import numpy as np

dataset_path = "dataset/50Ktrain.csv"
data = pd.read_csv(dataset_path)
# data = pd.read_csv(dataset_path)
print(data.shape)
# 篩選資料集，若有缺失值則整row刪除

keys_immutable = ["race", "sex", "native-country"]

# choose data[0] as input feature, for all data points, if the immutable feature of data point isn't same as data[0], then drop it
for key in keys_immutable:
    data = data[data[key] == data.iloc[0][key]]

# drop 'education' and 'fnlwgt' features
data = data.drop(
    ["education", "fnlwgt", "occupation", "race", "sex", "relationship"], axis=1
)

print(data.shape)

data = data.reset_index(drop=True)
print(data.head(3))
