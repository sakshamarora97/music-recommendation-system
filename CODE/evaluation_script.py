import numpy as np
import pandas as pd
import json
from sortedcontainers import SortedList
from sklearn.utils import shuffle
import ast
from sklearn.metrics import ndcg_score, dcg_score


# Loading data files final
def read_txt(file_name):
    return json.load(open(file_name))

song2user = read_txt("song2user.txt")
song2idx = read_txt("song2idx.txt")
user2idx = read_txt("user2idx.txt")
user2song = read_txt("user2song.txt")

track_features = read_txt("track_features_final.txt")

#Loading the user song rating dataframe
user_song_rating_csv_path = "user_song_interacting.csv"
df_overall = pd.read_csv(user_song_rating_csv_path).reset_index(drop = True)
df_overall.drop(columns = ["Unnamed: 0"], inplace=True)
df_overall_copy = df_overall.copy(deep = True)

song_ids_keep=[j for j in [song2idx.get(i['track_name'],"Not Found") for i in track_features] if j!="Not Found"]
df_overall=df_overall[df_overall["song_idx"].isin(song_ids_keep)]

usersong2tfidf = read_txt("user_song_tfidf.txt")
usersong2tfidf_new = {ast.literal_eval(key): val for key, val in usersong2tfidf.items()}
user_idx = pd.Series([key[0] for key, value in usersong2tfidf_new.items()])
song_idx = pd.Series([key[1] for key, value in usersong2tfidf_new.items()])
tf_idf = pd.Series([value for key, value in usersong2tfidf_new.items()])
cols = ['user_idx', 'song_idx','tf_idf']
df_usersong2tfidf = pd.DataFrame()
df_usersong2tfidf['user_idx'] = user_idx
df_usersong2tfidf['song_idx'] = song_idx
df_usersong2tfidf['tf_idf'] = tf_idf

df_overall = df_overall.merge(df_usersong2tfidf, on = ['user_idx','song_idx'],how = 'inner')

idx2song={v:k for k,v in song2idx.items()}
idx2user={v:k for k,v in user2idx.items()}

df_overall.loc[:,"user"]=df_overall["user_idx"].map(idx2user)
df_overall.loc[:,"track"]=df_overall["song_idx"].map(idx2song)

user_id_set=set(df_overall.user.values)
song_id_set=set(df_overall.track.values)
user2idx={}
song2idx={}
i=0
for user in user_id_set:
    user2idx[user]=i
    i+=1
for song in song_id_set:
    song2idx[song]=i
    i+=1

df_overall['user_id'] = df_overall.apply(lambda row: user2idx[row.user], axis=1)
df_overall['song_id'] = df_overall.apply(lambda row: song2idx[row.track], axis=1)

df_overall.drop(["user_idx","song_idx","user","track"],axis='columns',inplace=True)

df_overall=df_overall.rename(columns={'user_id':'user_idx','song_id':'song_idx'})[["user_idx","song_idx","Rating","tf_idf"]]

# a dictionary to tell us which users have rated which songs
user2song = {}
# a dicationary to tell us which songs have been rated by which users
song2user = {}
# a dictionary to look up ratings
usersong2rating = {}
print("Calling: update_user2song_and_song2user")
count = 0

def update_user2song_and_song2user(row):
    
    global count
    count += 1
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

    usersong2rating[(i,j)] = row.Rating

def update_usersong2rating_test(row):
    
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/len(df_test)))

    i = int(row.user_idx)
    j = int(row.song_idx)
    usersong2rating_test[(i,j)] = row.Rating

def predict(i, m):
    # calculate the weighted sum of deviations
    numerator = 0
    denominator = 0
    neighbors_get=neighbors.get(i,0)
    
    if neighbors_get!=0:
        for neg_w, j in neighbors.get(i):
            # remember, the weight is stored as its negative
            # so the negative of the negative weight is the positive weight
            try:
                numerator += -neg_w * deviations[j][m]
                denominator += abs(neg_w)
            except KeyError:
              # neighbor may not have rated the same rating
              # don't want to do dictionary lookup twice
              # so just throw exception
              pass

        if denominator == 0:
            prediction = averages.get(i)
        else:
            prediction = numerator / denominator + averages.get(i)
    else:
        prediction = averages.get(i,-1)
    if prediction==-1:
        #print("New User Identified")
        pass
    else:
        prediction = min(5, prediction)
        prediction = max(0.5, prediction) # min rating is 0.5
    return prediction

df_user_song_ct = df_overall.groupby(['user_idx']).agg(song_ct = ('song_idx', 'count')).reset_index()
# df_user_song_ct.head()

#***********Train Test Split**********
#Method1
df_overall["rank"] = df_overall.groupby("user_idx")["song_idx"].rank(method="dense", ascending=True)
df_overall['shuffled_rank'] = df_overall.groupby("user_idx")["rank"].transform(np.random.RandomState(seed=42).permutation)
df_join_counts = df_overall.merge(df_user_song_ct, on = 'user_idx', how = 'inner')
df_join_counts['idx_fraction'] = df_join_counts.shuffled_rank/df_join_counts.song_ct

split_fr = 0.8
df_join_counts['train_test_ind'] = df_join_counts['idx_fraction'].apply(lambda x: 'train' if x <split_fr else 'test')

data_train = df_join_counts[df_join_counts.train_test_ind == 'train']
data_test = df_join_counts[df_join_counts.train_test_ind == 'test']
data_train.apply(update_user2song_and_song2user,axis=1)

usersong2rating_test = {}
print("Calling: update_userrecipe2rating_test")
count = 0

data_test.apply(update_usersong2rating_test, axis=1)

#RUN COLLABORATIVE FILTERING ALGORITHM
K = 25 # number of neighbors we'd like to consider
limit = 5 # number of common recipes users must have in common in order to consider
neighbors = {} # store neighbors in this list
averages = {} # each user's average rating for later use
deviations = {} # each user's deviation for later use
SIGMA_CONST = 1e-6


for j1,i in enumerate(list(set(data_train.user_idx.values))):

    songs_i = user2song[i]
    songs_i_set = set(songs_i)

    # calculate avg and deviation
    ratings_i = { song:usersong2rating[(i, song)] for song in songs_i }
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = { song:(rating - avg_i) for song, rating in ratings_i.items() }
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    # save these for later use
    averages[i]=avg_i
    deviations[i]=dev_i

    sl = SortedList()

    for i1,j in enumerate(list(set(data_train.user_idx.values))):
        if j!=i:
            songs_j = user2song[j]
            songs_j_set = set(songs_j)
            common_songs = (songs_i_set & songs_j_set)
            if(len(common_songs)>limit):

                # calculate avg and deviation
                ratings_j = { song:usersong2rating[(j, song)] for song in songs_j }
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = { song:(rating - avg_j) for song, rating in ratings_j.items() }
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                # calculate correlation coefficient
                numerator = sum(dev_i[m]*dev_j[m] for m in common_songs)
                denominator = ((sigma_i+SIGMA_CONST) * (sigma_j+SIGMA_CONST))
                w_ij = numerator / (denominator)
                # insert into sorted list and truncate
                # negate absolute weight, because list is sorted ascending and we get all neighbors with the highest correlation
                # maximum value (1) is "closest"
                sl.add((-(w_ij), j))
                # Putting an upper cap on the number of neighbors
                if len(sl)>K:
                    del sl[-1]
    if i%100==0:                
        print((i,j1,sl))
    neighbors[i]=sl

train_predictions = []
train_targets = []
for (i, m), target in usersong2rating.items():
    
    # calculate the prediction for this recipe
    prediction = predict(i, m)

    # save the prediction and target
    train_predictions.append((i,m,prediction))
    train_targets.append((i,m,target))

test_predictions = []
test_targets = []
# same thing for test set
for (i, m), target in usersong2rating_test.items():
    
    # calculate the prediction for this recipe
    prediction_test = predict(i, m)

    # save the prediction and target
    test_predictions.append((i,m,prediction_test))
    test_targets.append((i,m,target))

# calculate accuracy
def mse(p, t):
    
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)

def rmse(p, t):
    
    p = np.array(p)
    t = np.array(t)
    return np.sqrt(np.mean((p - t)**2))

print('train mse:', mse([train_predictions[i][2] for i in range(len(train_predictions))], [train_targets[i][2] for i in range(len(train_targets))]))
print('test mse:', mse([test_predictions[i][2] for i in range(len(test_predictions))], [test_targets[i][2] for i in range(len(test_targets))]))

print('train rmse:', rmse([train_predictions[i][2] for i in range(len(train_predictions))], [train_targets[i][2] for i in range(len(train_targets))]))
print('test rmse:', rmse([test_predictions[i][2] for i in range(len(test_predictions))], [test_targets[i][2] for i in range(len(test_targets))]))


#Method2
np.random.seed(42)
df_shuffle=shuffle(df_overall)
split_fr = 0.8
cutoff = int(split_fr*len(df_shuffle))
data_train_v2 = df_shuffle.iloc[:cutoff]
data_test_v2 = df_shuffle.iloc[cutoff:]

# a dictionary to tell us which users have rated which songs
user2song = {}
# a dicationary to tell us which songs have been rated by which users
song2user = {}
# a dictionary to look up ratings

usersong2rating = {}
print("Calling: update_user2song_and_song2user")
count = 0

data_train_v2.apply(update_user2song_and_song2user,axis=1)

usersong2rating_test = {}
print("Calling: update_userrecipe2rating_test")
count = 0

data_test_v2.apply(update_usersong2rating_test, axis=1)

K = 25 # number of neighbors we'd like to consider
limit = 5 # number of common recipes users must have in common in order to consider
neighbors = {} # store neighbors in this list
averages = {} # each user's average rating for later use
deviations = {} # each user's deviation for later use
SIGMA_CONST = 1e-6


for j1,i in enumerate(list(set(data_train_v2.user_idx.values))):

    songs_i = user2song[i]
    songs_i_set = set(songs_i)

    # calculate avg and deviation
    ratings_i = { song:usersong2rating[(i, song)] for song in songs_i }
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = { song:(rating - avg_i) for song, rating in ratings_i.items() }
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    # save these for later use
    averages[i]=avg_i
    deviations[i]=dev_i

    sl = SortedList()

    for i1,j in enumerate(list(set(data_train_v2.user_idx.values))):
        if j!=i:
            songs_j = user2song[j]
            songs_j_set = set(songs_j)
            common_songs = (songs_i_set & songs_j_set)
            if(len(common_songs)>limit):

                # calculate avg and deviation
                ratings_j = { song:usersong2rating[(j, song)] for song in songs_j }
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = { song:(rating - avg_j) for song, rating in ratings_j.items() }
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                # calculate correlation coefficient
                numerator = sum(dev_i[m]*dev_j[m] for m in common_songs)
                denominator = ((sigma_i+SIGMA_CONST) * (sigma_j+SIGMA_CONST))
                w_ij = numerator / (denominator)
                # insert into sorted list and truncate
                # negate absolute weight, because list is sorted ascending and we get all neighbors with the highest correlation
                # maximum value (1) is "closest"
                sl.add((-(w_ij), j))
                # Putting an upper cap on the number of neighbors
                if len(sl)>K:
                    del sl[-1]
    if i%100==0:                
        print((i,j1,sl))
    neighbors[i]=sl

train_predictions = []
train_targets = []
for (i, m), target in usersong2rating.items():
    
    # calculate the prediction for this recipe
    prediction = predict(i, m)

    # save the prediction and target
    train_predictions.append((i,m,prediction))
    train_targets.append((i,m,target))

test_predictions = []
test_targets = []
# same thing for test set
for (i, m), target in usersong2rating_test.items():
    
    # calculate the prediction for this recipe
    prediction_test = predict(i, m)

    # save the prediction and target
    test_predictions.append((i,m,prediction_test))
    test_targets.append((i,m,target))

# calculate accuracy
def mse(p, t):
    
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)

def rmse(p, t):
    
    p = np.array(p)
    t = np.array(t)
    return np.sqrt(np.mean((p - t)**2))

print('train mse:', mse([train_predictions[i][2] for i in range(len(train_predictions))], [train_targets[i][2] for i in range(len(train_targets))]))
print('test mse:', mse([test_predictions[i][2] for i in range(len(test_predictions))], [test_targets[i][2] for i in range(len(test_targets))]))

print('train rmse:', rmse([train_predictions[i][2] for i in range(len(train_predictions))], [train_targets[i][2] for i in range(len(train_targets))]))
print('test rmse:', rmse([test_predictions[i][2] for i in range(len(test_predictions))], [test_targets[i][2] for i in range(len(test_targets))]))

#RANKING METRICS
df_ranking = df_overall.copy(deep = True)
df_ranking["truth_rank"] = df_ranking.groupby("user_idx")["tf_idf"].rank(method="dense", ascending=False)

userid_test_pred = pd.Series([test_predictions[i][0] for i in range(len(test_predictions))])
songid_test_pred = pd.Series([test_predictions[i][1] for i in range(len(test_predictions))])
pred_rating = pd.Series([test_predictions[i][2] for i in range(len(test_predictions))])

df_test_pred = pd.DataFrame()
df_test_pred['user_idx'] = userid_test_pred
df_test_pred['song_idx'] = songid_test_pred
df_test_pred['pred_rating'] = pred_rating
df_test_pred["pred_rank"] = df_test_pred.groupby("user_idx")["pred_rating"].rank(method="dense", ascending=False)

df_ranking_test = df_test_pred.merge(df_ranking, on = ['user_idx','song_idx'],how = 'inner')
df_ranking_test["truth_rank"] = df_ranking_test.groupby("user_idx")["tf_idf"].rank(method="dense", ascending=False)

#1. MAP@K Metric
df_map = df_ranking_test[['user_idx','song_idx','truth_rank','pred_rank']]
list_true = [list(df_map[df_map.user_idx == i]['truth_rank']) for i in list(set(df_map.user_idx))]
list_pred = [list(df_map[df_map.user_idx == i]['pred_rank']) for i in list(set(df_map.user_idx))]

def apk(actual, predicted, k):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return float(score)/ float(min(len(actual), k))

def mapk(actual, predicted, k):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
#     x = [apk(a,p,k) for a,p in zip(actual, predicted)]
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

#2. Mean NDCG Metric
def ndcg_metric(list_true,list_pred,k):
    try:
        true_relevance = np.asarray([list_true[k]])

        # Relevance scores in output order
        relevance_score = np.asarray([list_pred[k]])

        # DCG score
        dcg = dcg_score(true_relevance, relevance_score)
    #     print("DCG score : ", dcg)

        # IDCG score
        idcg = dcg_score(true_relevance, true_relevance)

        # Normalized DCG score
        ndcg = dcg / idcg
        # print("nDCG score : ", ndcg)
        return ndcg
    except:
        return np.nan    

ndcg = [ndcg_metric(list_true, list_pred, i) for i in range(len(list_true))]
avg_ndcg=pd.Series(ndcg)[~pd.Series(ndcg).isna()].mean()
