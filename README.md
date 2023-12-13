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
