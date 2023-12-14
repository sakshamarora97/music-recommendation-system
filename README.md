# music-recommendation-system
Course Project for CS 6242: Built a Tableau UI for song recommendation based on collaborative filtering with Python Backend and TabPy interface; used listening session histories of 1K users and 100k songs to create user-song interaction matrix

# Problem Definition
There are state of the art song recommendation systems available such as Spotify, Apple Music, Amazon
Music etc. but they lack in 2 key areas: 1) They fail to
provide any explainability on why a particular song
was recommended. 2) These recommendation system
do not provide users with tools to control the recommendations . Through our product, we aim to address
these challenges.

# Dataset

We used the [lastfm dataset](https://github.com/eifuentes/lastfm-dataset-1K). This dataset contains user, timestamp, artist, song tuples collected from Last.fm API, using the user.getRecentTracks() method.This dataset represents the whole listening habits (till May, 5th 2009) for nearly 1,000 users.

### Data Statistics

#### userid-timestamp-artid-artname-traid-traname.tsv

```
userid \t timestamp \t musicbrainz-artist-id \t artist-name \t musicbrainz-track-id \t track-name
```


| Element | Statistic |
| - | - |
| Total Lines | 19,150,868
| Unique Users | 992
| Artists with MBID | 107,528
| Artists without MBDID | 69,420

# Data Processing and Rating Creation
This data contains time based user-song interaction information. We
will utilize this information to create a user-song interaction rating matrix. Rating is a score in range 1-5 that
suggests how much a user likes the song. We developed
this rating by calculating monthly song frequency and
inverse song frequency. Monthly song frequency(TF)
is defined as the number of times a user has listened
to a song in a month and inverse song frequency(IDF)
is defined as a function of the number of users who
have listened to that song in that particular month. So
overall, the rating for that month is given as:

```math
Rating = TF_{Mt}*IDF_{Mt}
```
where
```math
\displaylines{IDF_{Mt} = \log(\frac{N}{1+df})\
N : \text{Total Number of users}\
df : \text{Number of users who have listened to that song in that particular month}}
```

This is done so that we capture niche user tastes in our song recommendation and songs that are listened by relatively less users is given higher weightage. Further, to capture temporal relationship, we give relatively higher weightage to recent songs i.e. most recent month is given weight as 1, second recent month is given weight as (23/24) and so on.

Further, we extracted following data-sets from Spo-
tify’s Web API: 1) New user’s song data, 2) MLHD
Song’s attributes, and 3) Top 500 songs of 2022 and
Top 500 songs of All time playlist’s songs and their
attributes


# Recommendation algorithm
## Collaborative Filtering
In this technique, we can base our algorithm either
on user-user or on song-song similarity. We identify
a set of closest neighbors for a given user i (identified through the Pearson correlation coefficient) based
on their ratings for common songs. Then, we take the
weighted average of the ratings that these neighboring users give to a song j in order to come up with a
predicted rating r(i,j) for a given user i for the song j.
Also , to account for user bias, we compute the same on
deviations of ratings around the mean for a given user.

# Data Workflow
The process involved taking the spotify user id as input  and getting data output for developing tableau dashboard. It can be divided into the following steps: 
1. Taking user Spotify user-id and pulling all their playlists and extracting all songs
2. Mapping user genre affinity.
3. Identify genre representative songs.
4. Creating user song matrix for a new user that is used as an input for collaborative filtering algorithm.
5. Generating personalised playlist using recommendation engine.
6. Generating recommendations from  "Top 500 songs of 2022" and "Top 500 songs of All Time" playlists by mapping ranked output to the mentioned playlist using cosine similarity. As user has intrinsic preferences towards a type of song, we exploited this assumption to map recommended songs to out of corpus songs using content based similarity.
   
The data pipeline produces 3 output files that are used as an input for Tableau dashboard.

<img width="550" alt="image" src="https://github.com/sakshamarora97/music-recommendation-system/assets/62840042/8cae8eeb-9cf2-4d98-b7c8-a7ece3485de1">


# Visualization and Product Features
The final product is a Tableau dashboard that uses Data workflow output. It shows the personalised recommended playlist and graphical representation of music attributes. Further, it provide tools to control the recommendations and dynamically update the playlist as per user's mood.

<img width="550" alt="image" src="https://github.com/sakshamarora97/music-recommendation-system/assets/62840042/da882153-9071-4460-b620-c12c67456a82">



The user interface has the following sections:
1. Input field to take user’s spotify id.
2. Playlist panel that has recommended songs along with their respective ranks
3. Graphical representation of recommended playlist’s song attributes.
4. Mood Control bar to control and change the recommendations as per user’s mood.
5. "How recommendation works?" button to describe and visualise how recommendations work.

The mood control section has the following toggles:
1. Choose Your Music: Selecting recommendation type from 3 options
  * Feeling Nostalgic
  * Best of 2022
  * Best of All Times
2. Danceability, Energy, Instrumentalness, Liveness, Song in Minutes: Length of a song in minutes.

<img width="100" alt="image" src="https://github.com/sakshamarora97/music-recommendation-system/assets/62840042/fc095a39-fcf7-49e6-81ab-a96bbd5fbc37">

The "How recommendation works?" button showcases the explainability aspect of the product. It visually explains the concept of collaborative filtering and visualizes user-neighbour interactions.

<img width="550" alt="image" src="https://github.com/sakshamarora97/music-recommendation-system/assets/62840042/b8bf2e7d-f6d6-49eb-9973-fb29dc40dcc6">


# Innovations
Summarizing the above sections, we introduced following innovation through our product : 
1. Defining rating matrix using song listening history of users.
2. Explainability aspect to recommendation i.e. explaining the attributes of the recommended playlist.
3. Interactive Mood controls giving users control over their recommendations. 4) Toggle to recommend songs from "Top 500 songs of 2022" and "Top 500 songs of All Time" playlists i.e. out of corpus recommendations.
4. Functionality to recommend songs to new users i.e. Spotify users not part of Lastfm dataset
5. Visualizing working of recommendation algorithm using user-neighbour interactions.

# Evaluation
## Ranking Evaluation:
### Metrics
### 1. Mean Average Precision@K:
This metric measures how many of the recommended results are relevant and are showing at the top.
Precision@K is the fraction of relevant items in the top K recommended results. The mean average precision@K measures the average precision@K averaged over all queries in the dataset.
```math
AP@k = \frac{1}{r_k}\sum_{k=1}^{k}s_k * rel(k)/k,
```
where $s_k$ is the number of relevant songs in top k results, 
```math 
rel(k) = 1
```
if kth songs are relevant and 0 otherwise
```math
mAP@k = \frac{1}{N}\sum_{i=1}^{N}AP@k
```


2. Normalized Discounted Cumulative Gain:
Gain refers to the relevance score for each item (song in this case) recommended. Cumulative gain at K is the sum of gains of the first K items recommended. Discounted cumulative gain (DCG) weighs each relevance score based on its position, with a higher weight to the recommendations at the top. 

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$
where
$\displaystyle{DCG@K = \sum_{i=1}^{K} \frac{G_i}{log_2{(i+1)}}}$ 
and
$\displaystyle{IDCG@K = \sum_{i=1}^{K_i} \frac{G_i}{log_2{(i+1)}}}$, 
where $K_i$ is K(ideal) and $G_i$ is G (relevance score of each recommended item). 

### Results:
We followed two different approaches to sample our dataset into 80\%-20\% train-test split. The first approach takes a stratified sampling route where for each user the split is done randomly into 80-20 split. The second approach does an overall random shuffling and we obtain the train (80\%) and test (20\%).

<img width="300" alt="image" src="https://github.com/sakshamarora97/music-recommendation-system/assets/62840042/ff9ad1e4-1886-4d4e-ab4e-7f338d8ef780">



## User Feedback

We conducted user survey to seek feedback on our
product. We got 17 responses and the overall response
5
was positive. One promising observation has been a
strong positive response for "Explainability Visualiza-
tion" from respondents who did not care about explain-
ability in the first place

<img width="300" alt="image" src="https://github.com/sakshamarora97/music-recommendation-system/assets/62840042/c6bdc5ee-a139-4ae4-be9a-8243dfa3f7c9">


