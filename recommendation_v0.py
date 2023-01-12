import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList
import numpy.linalg as ln
import json
# import new_user_integration
import ast
import math
from Explore import get_new_user_songs as ns
from sklearn.metrics.pairwise import cosine_similarity
t1 = datetime.now()
f = open("log_for_data_pull.txt", "a")
f.write("Initiating Run!"+"{}".format(t1)+"\n")
f.close()


def user_input(spotify_user_id):

    if spotify_user_id =='Reset':
        return "Enter Spotify User-id"
    else: pass

    user_name = "Adam"

    def get_direction(row):
        return str(row.user_id)+'-->'+str(row.neighbor_id)

    def min_max_scaling(df):
        # copy the dataframe
        df_norm = df.copy()
        # apply min-max scaling
        for column in features:
            df_norm[column] = (df_norm[column] - df_norm[column].min()) / \
                (df_norm[column].max() - df_norm[column].min())

        return df_norm

    def recommendation_mapping_2022(billboard_song_feature_dataframe, collab_song_feature_dataframe):
        features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
                    'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
        features = [i.lower() for i in features]
        a = billboard_song_feature_dataframe[features]
        b = collab_song_feature_dataframe[features]
    #     a = (a-a.mean())/a.std()
    #     b = (b-a.mean())/a.std()
        a.to_numpy()
        b.to_numpy()
        billboard_songs = billboard_song_feature_dataframe
        cosine = cosine_similarity(a, b)
        cosine = cosine.reshape(cosine.shape[0],)
        billboard_songs.loc[:, "cosine_similarity"] = pd.Series(cosine)
        billboard_songs = billboard_songs.sort_values(
            by=['cosine_similarity'], ascending=False)
        temp = pd.DataFrame()
        temp = pd.concat([temp, billboard_songs.iloc[0]])
        temp = temp.transpose()
        billboard_songs = billboard_songs.reset_index(drop=True)
        billboard_songs = billboard_songs.drop(columns='cosine_similarity')

        return temp

    def corpus_song_mapping(corpus_song_feature_dataframe, user_average_song_feature_dataframe):
        """
        Input: 
        Corpus_song_feature_dataframe: Pandas Dataframe (N X D2)
        Use_average_song_feature_dataframe: Pandas Dataframe (1 X D1)


        Output:
        Genre based Rank ordered Corpus Song Feature Dataframe
        """
        features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
                    'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
        features = [i.lower() for i in features]
        # filtering variables to be used for cosine similairty
        a = corpus_song_feature_dataframe[features]
        # a.apply(lambda x:(x-x.mean())/x.std(),axis=0)x
        a = (a-a.mean())/a.std()
    #     print(a)
        b = user_average_song_feature_dataframe[features]
        #b = b.apply(lambda x:(x-x.mean())/x.std(),axis=0)
        b = (b-a.mean())/a.std()
    #     print(b)
        a.to_numpy()
        b.to_numpy()
        # Calculating cosine similarity of user song with all corpus songs and sorting it in descending order

        genre = user_average_song_feature_dataframe["genres"]

        df_subset = corpus_song_feature_dataframe[corpus_song_feature_dataframe["genres"].apply(
            lambda x: bool(1) if x.count(genre) else bool(0))]
    #     print(df_subset.s)

        try:
            a = df_subset[features]
            a = (a-a.mean())/a.std()
            b = (b-a.mean())/a.std()
            a.to_numpy()
            b.to_numpy()
            cosine = np.dot(a, b)/(ln.norm(a, axis=1)*ln.norm(b))
            df_subset.loc[:, 'cosine_similarity'] = cosine
    #         print(df_subset)
    #         return df_subset.sort_values(by=['cosine_similarity'], ascending= False).head(k)["id"].reset_index(drop=True)[0:k]
            return list(df_subset.sort_values(by=['cosine_similarity'], ascending=False).head(5)["id"].reset_index(drop=True)[:])
        except:
            a = corpus_song_feature_dataframe[features]
            a = (a-a.mean())/a.std()
            b = (b-a.mean())/a.std()
            a.to_numpy()
            b.to_numpy()
            cosine = np.dot(a, b)/(ln.norm(a, axis=1)*ln.norm(b))
            corpus_song_feature_dataframe.loc[:, 'cosine_similarity'] = cosine
    #         return df_subset.sort_values(by=['cosine_similarity'], ascending= False).head(k)["id"].reset_index(drop=True)[0:k]
            return list(corpus_song_feature_dataframe.sort_values(by=['cosine_similarity'], ascending=False).head(5)["id"].reset_index(drop=True)[:])

    def update_user2song_and_song2user(row):

        # global count
        # count += 1
        #     if count % 100000 == 0:
        #         print("processed: %.3f" % (float(count)/cutoff))

        i = int(row.user_idx)
        j = int(row.song_idx)
        if i not in user2song:
            user2song[i] = [j]
        else:
            user2song[i].append(j)

        if j not in song2user:
            song2user[j] = [i]
        else:
            song2user[j].append(i)

        usersong2rating[(i, j)] = row.Rating

    def get_songs_recommendations(i):
        common_songs_dict = {}
        songs_i = user2song[i]
        songs_i_set = set(songs_i)

        # calculate avg and deviation
        ratings_i = {song: usersong2rating[(i, song)] for song in songs_i}
        avg_i = np.mean(list(ratings_i.values()))
        dev_i = {song: (rating - avg_i) for song, rating in ratings_i.items()}
        dev_i_values = np.array(list(dev_i.values()))
        sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

        # save these for later use
        averages[i] = avg_i
        deviations[i] = dev_i

        sl = SortedList()
        for j in list(set(df_overall.user_idx.values)):
            if j != i:
                songs_j = user2song[j]
                songs_j_set = set(songs_j)
                common_songs = (songs_i_set & songs_j_set)

                if (len(common_songs) > limit):
                    common_songs_dict[j] = list(common_songs)
                    ratings_j = {song: usersong2rating[(
                        j, song)] for song in songs_j}
                    avg_j = np.mean(list(ratings_j.values()))
                    dev_j = {song: (rating - avg_j)
                             for song, rating in ratings_j.items()}
                    dev_j_values = np.array(list(dev_j.values()))
                    sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                    # calculate correlation coefficient
                    numerator = sum(dev_i[m]*dev_j[m] for m in common_songs)
                    denominator = ((sigma_i+SIGMA_CONST)
                                   * (sigma_j+SIGMA_CONST))
                    w_ij = numerator / (denominator)
                    # insert into sorted list and truncate
                    # negate absolute weight, because list is sorted ascending and we get all neighbors with the highest correlation
                    # maximum value (1) is "closest"
                    sl.add((-np.abs(w_ij), j))
                    # Putting an upper cap on the number of neighbors
                    if len(sl) > K:
                        del sl[-1]

        neighbors[i] = sl
        try:
            most_related_neighbors = [j for i, j in neighbors[i][:10]]
        except:
            most_related_neighbors = [j for i, j in neighbors[i]]
        for i in most_related_neighbors:
            songs_i = user2song[i]
            songs_i_set = set(songs_i)
            ratings_i = {song: usersong2rating[(i, song)] for song in songs_i}
            recommended_song_list.append(ratings_i)

        total = Counter()
        for j in recommended_song_list:
            total += Counter(j)
        recommended_song_ids = [i for i, j in total.most_common(500)]
        recommended_songs = []

        for song_id in recommended_song_ids:
            recommended_songs.append(
                [(i, j) for i, j in song2idx.items() if j == song_id][0][0])
        recommended_songs = list(set(recommended_songs)-set(user2song[i]))

        return most_related_neighbors, recommended_songs, common_songs_dict

    def find_commmon(row):
        common_songs = common_songs_dict[row.nearest_neighbors]
        if row.songs in common_songs:
            return "Common"
        return "Not Common"

    # List of Song Features
    features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
                'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    features = [i.lower() for i in features]

    # Loading in the necessary mapper files created
    song2idx = json.load(open("song2idx.txt"))
    user2idx = json.load(open("user2idx.txt"))

    idx2song = {v: k for k, v in song2idx.items()}
    idx2user = {v: k for k, v in user2idx.items()}

    # Loading in the set of all tracks listened by atleast 5 people along with the song features
    track_features = json.load(open("track_features.txt"))
    f = open("log_for_data_pull.txt", "a")
    f.write("Tracks with Song Features loaded!"+"\n")
    f.close()

    print("Tracks with Song Features loaded")

    df_song_features = pd.read_csv("df_song_features.csv")
    u, v = df_song_features.shape
    print(f"Shape of Track Features Dataset : {u,v}")

    df_song_features.drop_duplicates(
        subset=df_song_features.columns.difference(['genres']), inplace=True)

    df_song_features.drop(columns='Unnamed: 0', axis=0, inplace=True)
    df_song_features[["genres"]] = df_song_features[[
        "genres"]].fillna("").apply(list)

    # Loading in the User-Song Rating dataset based on TF-IDF values
    df_overall = pd.read_csv("user_song_interacting.csv", index_col=False)
    df_overall.drop(columns='Unnamed: 0', axis=0, inplace=True)

    song_ids_keep = [j for j in [song2idx.get(
        i['track_name'], "Not Found") for i in track_features] if j != "Not Found"]
    df_overall = df_overall[df_overall["song_idx"].isin(song_ids_keep)]

    u, v = df_overall.shape
    f = open("log_for_data_pull.txt", "a")
    f.write("Now the file has more content!"+"\n")
    f.close()

    print(
        f"Shape of User-Song Interaction Data filtered with songs heard by atleast 5 people : {u,v}")

    df_overall.loc[:, "user"] = df_overall["user_idx"].map(idx2user)
    df_overall.loc[:, "track"] = df_overall["song_idx"].map(idx2song)

    # Updating the user2idx and song2idx dicts as the Overall Dataframe is filtered

    user_id_set = set(df_overall.user.values)
    song_id_set = set(df_overall.track.values)
    user2idx = {}
    song2idx = {}
    i = 0
    for user in user_id_set:
        user2idx[user] = i
        i += 1
    for song in song_id_set:
        song2idx[song] = i
        i += 1

    new_id_track_mapping = {v: k for k, v in song2idx.items()}
    df_overall['user_id'] = df_overall.apply(
        lambda row: user2idx[row.user], axis=1)
    df_overall['song_id'] = df_overall.apply(
        lambda row: song2idx[row.track], axis=1)

    df_overall.drop(["user_idx", "song_idx", "user", "track"],
                    axis='columns', inplace=True)

    df_overall = df_overall.rename(columns={'user_id': 'user_idx', 'song_id': 'song_idx'})[
        ["user_idx", "song_idx", "Rating"]]

    # Creating a rating input for a new user

    df_song_genre = ns.main(spotify_user_id)

    df_song_genre.loc[:, "Rating"] = pd.cut(
        df_song_genre["num_tracks_in_genre"].rank(pct=True), bins=5, labels=[1, 2, 3, 4, 5])
    song_ids = pd.Series(df_song_genre.apply(
        lambda x: corpus_song_mapping(df_song_features, x), axis=1))
    df_song_genre.loc[:, 'song_ids'] = song_ids
    song_ids = list(set(song_ids.explode()))
    track_details = df_song_features[df_song_features["id"].isin(song_ids)][[
        "id", "track_name"]]
    id_to_track_mapping = dict(zip(track_details.id, track_details.track_name))
    df_song_genre.loc[:, "track_name"] = df_song_genre.apply(
        lambda x: [id_to_track_mapping[i] for i in x["song_ids"]], axis=1)
    df_song_genre.loc[:, "old_ids"] = df_song_genre.apply(
        lambda x: [song2idx[i] for i in x["track_name"]], axis=1)

    # Creating DataFrame for the input user to integrate into user-song interaction matrix

    df_new_user = df_song_genre[["old_ids", "Rating"]]
    df_new_user = df_new_user.explode('old_ids')
    df_new_user = df_new_user.groupby(["old_ids"], as_index=False).max()
    df_new_user.loc[:, "user_idx"] = df_overall["user_idx"].max()+1
    df_new_user.rename(columns={"old_ids": "song_idx"}, inplace=True)
    df_new_user = df_new_user[["user_idx", "song_idx", "Rating"]]

    f = open("log_for_data_pull.txt", "a")
    f.write("New user Data created in the required format!"+"\n")
    f.close()

    print("New user Data created in the required format")
    u, v = df_new_user.shape
    print(f"Shape of the new user dataframe : {u,v}")

    # Appending the new user data into our original data
    df_overall = df_overall.append(df_new_user, ignore_index=True)

    # a dictionary to tell us which users have rated which songs
    user2song = {}
    # a dicationary to tell us which songs have been rated by which users
    song2user = {}
    # a dictionary to look up ratings
    usersong2rating = {}
    print("Calling: update_user2song_and_song2user")
    # count = [0]

    # # Appending the new user data into our original data
    # df_overall = df_overall.append(df_new_user, ignore_index=True)

    # # a dictionary to tell us which users have rated which songs
    # user2song = {}
    # # a dicationary to tell us which songs have been rated by which users
    # song2user = {}
    # # a dictionary to look up ratings
    # usersong2rating = {}
    # print("Calling: update_user2song_and_song2user")
    # count = 0

    df_overall.apply(update_user2song_and_song2user, axis=1)

    # Creating user-user Collbarative filtering algorithm

    K = 25  # number of neighbors we'd like to consider
    limit = 1  # number of common songs users must have in common in order to consider
    neighbors = {}  # store neighbors in this list
    averages = {}  # each user's average rating for later use
    deviations = {}  # each user's deviation for later use
    SIGMA_CONST = 1e-6
    recommended_song_list = []

    user_id = df_overall["user_idx"].max()
    nearest_neighbors, recommended_songs, common_songs_dict = get_songs_recommendations(
        user_id)
    f = open("log_for_data_pull.txt", "a")
    f.write("Recommendation ranking created through collaborative filtering!"+"\n")
    f.close()

    # Preparing data for UI
    df_user_user_ui = pd.DataFrame(
        {'user_name': user_name, 'user_id': user_id, 'nearest_neighbors': pd.Series(nearest_neighbors)})
    df_user_user_ui.loc[:, "songs"] = df_user_user_ui["nearest_neighbors"].map(
        user2song)
    df_user_user_ui = df_user_user_ui.explode('songs')
    df_user_user_ui.loc[:, "common_song"] = df_user_user_ui.apply(
        lambda x: find_commmon(x), axis=1)
    df_user_user_ui.loc[:, "rating"] = df_user_user_ui.apply(
        lambda x: usersong2rating.get((x.user_id, x.songs), "Not Rated yet"), axis=1)

    f = open("log_for_data_pull.txt", "a")
    f.write("Creating an new user id to track mapping for further use!"+"\n")
    f.close()

    print("Creating an new user id to track mapping for further use")
    # new_id_track_mapping = {}
    # for v, k in enumerate(list(df_user_user_ui["songs"].unique())):
    #     if v % 500 == 0:
    #         print(f"{v} songs done")
    #     new_id_track_mapping[k] = [i for i, j in song2idx.items() if j == k][0]

    df_user_user_ui.loc[:, "track_name"] = df_user_user_ui["songs"].map(
        new_id_track_mapping)

    df_user_user_ui = df_user_user_ui.rename(columns={"songs": "song_id"})

    df_feature_subset = df_song_features[df_song_features["track_name"].isin(list(new_id_track_mapping.values()))][['track_name', 'genres', 'danceability',
                                                                                                                    'energy',
                                                                                                                    'key',
                                                                                                                    'loudness',
                                                                                                                    'mode',
                                                                                                                    'speechiness',
                                                                                                                    'acousticness',
                                                                                                                    'instrumentalness',
                                                                                                                    'liveness',
                                                                                                                    'valence',
                                                                                                                    'tempo',
                                                                                                                    'duration_ms', "time_signature"]]

    df_user_user_final = df_user_user_ui.merge(
        df_feature_subset, how="inner", on=["track_name"])

    df_user_user_common = df_user_user_final[df_user_user_final['common_song'] == 'Common']
    df_user_user_common.loc[:, "common_songs"] = df_user_user_common["nearest_neighbors"].map(
        {k: len(v) for k, v in common_songs_dict.items()})
    df_user_user_common = df_user_user_common[['user_name', 'user_id', 'nearest_neighbors', 'song_id', 'common_songs', 'rating', 'track_name', 'danceability', 'energy', 'key',
                                               'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                                               'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']].reset_index(drop=True)
    df_user_user_common["user_id"] = 'u0'
    neighbor_mapping = {}
    for i, j in enumerate(df_user_user_common["nearest_neighbors"].unique()):
        neighbor_mapping[j] = 'u'+str(i+1)
    df_user_user_common.loc[:, "neighbor_id"] = df_user_user_common["nearest_neighbors"].map(
        neighbor_mapping)
    df_user_user_common = df_user_user_common[['user_name', 'user_id', 'neighbor_id', 'song_id', 'common_songs', 'rating', 'track_name', 'danceability', 'energy', 'key',
                                               'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                                               'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']].reset_index(drop=True)

    # Using polar coordinate system
    r = np.arange(1, 2, 0.1)
    theta = np.arange(0, 2*math.pi, 0.2*math.pi)
    user_x_mapping = {}
    user_y_mapping = {}
    for i in range(10):
        user_x_mapping["u"+str(i+1)] = r[i]*math.cos(theta[i])
        user_y_mapping["u"+str(i+1)] = r[i]*math.sin(theta[i])

    df_user_user_common.loc[:, "X_coordinate"] = df_user_user_common["neighbor_id"].map(
        user_x_mapping)
    df_user_user_common.loc[:, "Y_coordinate"] = df_user_user_common["neighbor_id"].map(
        user_y_mapping)

    df_user_user_common.loc[:, "direction"] = df_user_user_common.apply(
        lambda x: get_direction(x), axis=1)
    df_user_user_common.loc[:, "base"] = 1

    df_user_user_v2 = df_user_user_common.copy()
    df_user_user_common[['neighbor_id', 'user_id']
                        ] = df_user_user_common[['user_id', 'neighbor_id']]
    df_user_user_common["base"] = 2
    df_user_user_v2 = df_user_user_v2.append(df_user_user_common)
    df_user_user_v2.loc[:, "neighbor_name"] = pd.Series(np.where(df_user_user_v2["base"] == 1, df_user_user_v2["neighbor_id"].apply(
        lambda x: x.upper()), df_user_user_v2["user_id"].apply(lambda x: x.upper())))
    df_user_user_v2.loc[:, "X"] = np.where(df_user_user_v2["base"] == 1, 0, 0)
    df_user_user_v2.loc[:, "Y"] = np.where(df_user_user_v2["base"] == 1, 0, 0)

    df_user_user_v2 = min_max_scaling(df_user_user_v2)

    df_user_user_v2.to_csv("user-user_v2.csv")

    f = open("log_for_data_pull.txt", "a")
    f.write("csv exported for UI vizualization!"+"\n")
    f.close()

    print("csv exported for UI vizualization")

    u, v = df_user_user_v2.shape
    print(f"Shape of User-User Data for UI : {u,v}")

    # Ranking the Recommended Songs
    rank_recommendations = {}
    for i, j in enumerate(recommended_songs):
        rank_recommendations[j] = i+1
    df_recommendation = df_song_features.loc[df_song_features["track_name"].isin(
        recommended_songs)]
    df_recommendation = df_recommendation.drop_duplicates(subset=[
                                                          "track_name"])
    df_recommendation.loc[:, "Ranking"] = df_recommendation["track_name"].map(
        rank_recommendations)
    df_recommendation = df_recommendation[['track_name', 'genres', 'Ranking', 'danceability',
                                           'energy',
                                           'key',
                                           'loudness',
                                           'mode',
                                           'speechiness',
                                           'acousticness',
                                           'instrumentalness',
                                           'liveness',
                                           'valence',
                                           'tempo',
                                           'duration_ms']]

    df_recommendation.loc[:,
                          "song_length_mins"] = df_recommendation["duration_ms"]/60000
    df_recommendation.loc[:, "song_category"] = "Old Songs"
    df_recommendation = df_recommendation.reset_index(drop=True)

    # Mapping the Recommended songs to Top songs of 2022 and Top overall songs

    df_top_2022 = pd.read_csv("spotify_top500_2022.csv")
    df_top_overall = pd.read_csv("top500_all_time.csv")
    df_top_2022.drop(columns='Unnamed: 0', axis=0, inplace=True)
    df_top_overall.drop(columns='Unnamed: 0', axis=0, inplace=True)
    for i in range(df_top_2022.shape[0]):
        if i == 0:
            df_feature_subset = pd.DataFrame(ast.literal_eval(
                df_top_2022["Song Attributes"][i]), index=[0])
        else:
            df_feature_subset = pd.concat([df_feature_subset, pd.DataFrame(
                ast.literal_eval(df_top_2022["Song Attributes"][i]), index=[0])])

    for i in range(df_top_overall.shape[0]):
        if i == 0:
            df_feature_subset_overall = pd.DataFrame(ast.literal_eval(
                df_top_overall["Song Attributes"][i]), index=[0])
        else:
            df_feature_subset_overall = pd.concat([df_feature_subset_overall, pd.DataFrame(
                ast.literal_eval(df_top_overall["Song Attributes"][i]), index=[0])])

    df_feature_subset_overall = df_feature_subset_overall[['danceability',
                                                           'energy',
                                                           'key',
                                                           'loudness',
                                                           'mode',
                                                           'speechiness',
                                                           'acousticness',
                                                           'instrumentalness',
                                                           'liveness',
                                                           'valence',
                                                           'tempo',
                                                           'type',
                                                           'duration_ms',
                                                           'time_signature']]
    df_feature_subset = df_feature_subset[['danceability',
                                           'energy',
                                           'key',
                                           'loudness',
                                           'mode',
                                           'speechiness',
                                           'acousticness',
                                           'instrumentalness',
                                           'liveness',
                                           'valence',
                                           'tempo',
                                           'type',
                                           'duration_ms',
                                           'time_signature']]

    df_top_2022.rename(columns={'Genre List': 'genres'}, inplace=True)
    df_top_overall.rename(columns={'Genre List': 'genres'}, inplace=True)
    df_feature_subset.reset_index(inplace=True)
    df_feature_subset_overall.reset_index(inplace=True)
    df_top_2022 = pd.concat([df_top_2022, df_feature_subset], axis=1)
    df_top_overall = pd.concat(
        [df_top_overall, df_feature_subset_overall], axis=1)

    top_2022_reco = pd.DataFrame(columns=df_top_2022.columns)
    top_overall_reco = pd.DataFrame(columns=df_top_overall.columns)
    for i in range(df_recommendation.shape[0]):
        top_2022_reco = pd.concat([top_2022_reco, recommendation_mapping_2022(
            df_top_2022, df_recommendation[df_recommendation.index == i])])
    for i in range(df_recommendation.shape[0]):
        top_overall_reco = pd.concat([top_overall_reco, recommendation_mapping_2022(
            df_top_overall, df_recommendation[df_recommendation.index == i])])

    top_2022_reco = top_2022_reco.rename(columns={'Track Name': 'track_name'})
    top_2022_reco.loc[:,
                      "song_length_mins"] = top_2022_reco["duration_ms"]/60000
    top_2022_reco.loc[:, "song_category"] = "New Songs 2022"
    top_2022_reco = top_2022_reco.reset_index(drop=True)
    top_2022_reco["Ranking"] = pd.Series(
        np.arange(1, len(df_recommendation)+1))

    top_overall_reco = top_overall_reco.rename(
        columns={'Track Name': 'track_name'})
    top_overall_reco.loc[:,
                         "song_length_mins"] = top_overall_reco["duration_ms"]/60000
    top_overall_reco.loc[:, "song_category"] = "Overall New Songs"
    top_overall_reco = top_overall_reco.reset_index(drop=True)
    top_overall_reco["Ranking"] = pd.Series(
        np.arange(1, len(df_recommendation)+1))

    df_recommendation_2022 = top_2022_reco[df_recommendation.columns]
    df_recommendation_alltime = top_overall_reco[df_recommendation.columns]
    df_recommendation_overall = pd.concat(
        [df_recommendation, df_recommendation_2022])
    df_recommendation_overall = pd.concat(
        [df_recommendation_overall, df_recommendation_alltime])
    df_recommendation_overall = df_recommendation_overall.reset_index(
        drop=True)

    features = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
                'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    features = [i.lower() for i in features]

    df_recommendation_overall = min_max_scaling(df_recommendation_overall)

    df_recommendation_overall.to_csv("recommendations_overall.csv")
    u, v = df_recommendation_overall.shape

    f = open("log_for_data_pull.txt", "a")
    f.write("Recommendation output csv exported for UI vizualization"+"\n")
    f.close()

    print(f"Shape of my Recommendation dataset is :{u,v}")
    f = open("log_for_data_pull.txt", "a")
    f.write("run completed!"+"{}".format(datetime.now())+"\n")
    f.close()
    return "Recommendations Generated! Refresh View!!"

