import pandas as pd
import numpy as np
from collections import Counter
import math
import json


df = pd.read_csv("userid-timestamp-artid-artname-traid-traname.tsv", sep='\t', header=None,
                 names=[
                     'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name'
                 ],
                 skiprows=[
                     2120260-1, 2446318-1, 11141081-1,
                     11152099-1, 11152402-1, 11882087-1,
                     12902539-1, 12935044-1, 17589539-1
                 ], parse_dates=["timestamp"])

print(df.head())

df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month

df.groupby(["year"])[["user_id"]].count()
print("Major portion of the data is from 3 years (2006-2008) and hence we will use these 3 years")

df = df.loc[df.year.isin([2006, 2007, 2008])].reset_index(drop=True)
df = df.loc[~df["track_name"].isna()].reset_index(drop=True)


user2idx = {}
song2idx = {}
# count=0
user_set = set(df.user_id.values)
song_set = set(df.track_name.values)

count = 0
for i in user_set:
    user2idx[i] = count
    count += 1
user2idx

count = 0
for i in song_set:
    song2idx[i] = count
    count += 1

df["user_idx"] = df.apply(lambda x: user2idx[x.user_id], axis=1)
df["song_idx"] = df.apply(lambda x: song2idx[x.track_name], axis=1)

#2006,2007,2008

df.sort_values(by=['year','month'])[['year','month']].drop_duplicates()

max_year = df['year'].max()
max_month = 12
df
df.loc[:, "freq"] = 1

df_subset = df.groupby(["user_idx", "song_idx", "year", "month"], as_index=False).agg(
    {"freq": "sum"}).assign(rel_weight=lambda x: 1-((12*(max_year-x.year) + (max_month-x.month))/36))

df_subset.loc[:, "weighted_freq"] = df_subset["freq"]*df_subset["rel_weight"]
df_interaction = df_subset.groupby(["user_idx", "song_idx"], as_index=False).agg({
    "freq": 'sum', "weighted_freq": "sum"})


print("Computing Inverse Song Frequency")

df_1 = df_subset.groupby(["year", "month"], as_index=False)[
    "user_idx"].nunique().rename(columns={"user_idx": "n_users"})
df_2 = df_subset.groupby(["year", "month", "song_idx"], as_index=False)[
    "user_idx"].nunique().rename(columns={"user_idx": "df"})
df_merge = df_2.merge(df_1, how="inner", on=["year", "month"])

df_merge["inverse_song_freq"] = df_merge.apply(
    lambda x: math.log((x.n_users/(1+x.df)), 2), axis=1)

print("Computing TF-IDF")
df_final = df_subset.merge(df_merge, how="inner", on=[
                           "year", "month", "song_idx"])

df_final.loc[:, "tf-idf"] = df_final["weighted_freq"] * \
    df_final["inverse_song_freq"]

df_final = df_final.groupby(["user_idx", "song_idx"], as_index=False)[
    "tf-idf"].sum()

df_final.shape

# a dictionary to look up ratings
usersong2tfidf = {}
print("Calling: update_user2song_and_song2user")
count = 0


def update_user2song_and_song2user(row):

    global count
    count += 1
#     if count % 100000 == 0:
#         print("processed: %.3f" % (float(count)/cutoff))

    i = int(row.user_idx)
    j = int(row.song_idx)

    usersong2tfidf[(i, j)] = row["tf-idf"]


df_final.apply(update_user2song_and_song2user, axis=1)

usersong2tfidf_new = {str(key): val for key, val in usersong2tfidf.items()}
with open('user_song_tfidf.txt', 'w') as convert_file:
    convert_file.write(json.dumps(usersong2tfidf_new))
