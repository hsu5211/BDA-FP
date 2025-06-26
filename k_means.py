import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

public_df = pd.read_csv('public_data.csv')
private_df = pd.read_csv('private_data.csv')

for name, df in [('public', public_df), ('private', private_df)]:
    print(f'[{name}] missing value check:')
    print(df.isnull().sum())

public_features = public_df[['1', '2', '3', '4']]
private_features = private_df[['1', '2', '3', '4', '5', '6']]

public_scaler = StandardScaler()
public_scaled = public_scaler.fit_transform(public_features)

private_scaler = StandardScaler()
private_scaled = private_scaler.fit_transform(private_features)

print('[public] execute KMeans clustering (15 clusters)...')
public_kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
public_labels = []
for i in tqdm(range(len(public_scaled)), desc='[public] KMeans clustering'):
    if i == 0:
        public_labels = public_kmeans.fit_predict(public_scaled)
        break

print('[private] execute KMeans clustering (23 clusters)...')
private_kmeans = KMeans(n_clusters=23, random_state=42, n_init=10)
private_labels = []
for i in tqdm(range(len(private_scaled)), desc='[private] KMeans clustering'):
    if i == 0:
        private_labels = private_kmeans.fit_predict(private_scaled)
        break

public_out = pd.DataFrame({'id': range(1, len(public_labels)+1), 'label': public_labels})
private_out = pd.DataFrame({'id': range(1, len(private_labels)+1), 'label': private_labels})

public_out.to_csv('public_submission.csv', index=False)
private_out.to_csv('private_submission.csv', index=False)

print('Output public_submission.csv and private_submission.csv')
