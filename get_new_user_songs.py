import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import stats as st


client_credentials_manager = SpotifyClientCredentials(client_id = '3c59905bf8b14112af71062075bb919c', 
                                                      client_secret = '179b13cf995a4f548cd43e6988d9f7d6')

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_playlists(user_id):
    
    """Gets all playlist ids for a given user_id. Takes care of pagination using next

    Args:
        user_id (string): Spotify user_id 

    Returns:
        list : List of ids of all public playlist of user
    """
    playlists_paginated = sp.user_playlists(user_id)
    all_playlists_ids = []
    while playlists_paginated:
        for playlist in playlists_paginated['items']:
            all_playlists_ids.append(playlist['id'])
        if playlists_paginated['next']:
            playlists_paginated = sp.next(playlists_paginated)
        else:
            playlists_paginated = None
    return all_playlists_ids




def get_playlist_tracks(playlist_id):
    """Gets all track_ids in a given playlist

    Args:
        playlist_id (string): Spotify playlist_id

    Returns:
        list,list: List of ids of all tracks & all artists in playlist
    """
    #check for episode pending
    tracks_paginated = sp.playlist_items(playlist_id = playlist_id)
    all_track_ids = []
    all_artist_ids = []
    while tracks_paginated:
        for track in tracks_paginated['items']:
            if track["track"] is None:
                continue
            all_track_ids.append(track["track"]['id'])
            all_artist_ids.append(track["track"]["artists"][0]["id"])
        if tracks_paginated['next']:
            tracks_paginated = sp.next(tracks_paginated)
        else:
            tracks_paginated = None
    return all_track_ids,all_artist_ids



def get_genre_for_track(track_artist):
    """Get track genres given track_id

    Args:
        track_id (string): Spotify track id

    Returns:
        list: List of Genres of track
    """
    return sp.artist(track_artist)['genres']


def get_user_tracks(user_id):
    """Given a user, pull all tracks and their audio features across all their playlists

    Args:
        users (list): list of users

    Returns:
        pd.DataFrame, pd.DataFrame: User song dataframe, track information dataframe
    """
    new_users_info = pd.DataFrame({"user_id":[],"user_name":[],"playlist_id":[],"playlist_name":[],"track_id":[],"track_name":[],"artist_id":[],"genres":[]})
    i=0
    j=0

    tracks_info = pd.DataFrame({"track_id":[],"artist_id":[],"genres":[],"danceability":[],"energy":[],"key":[],"loudness":[],"mode":[],"speechiness":[],"acousticness":[],"instrumentalness":[],"liveness":[],"valence":[],"tempo":[],"track_href":[],"duration_ms":[],"time_signature":[]})
    user_name = sp.user(user_id)['display_name']
    print("Pulling data for User: ",user_name)
    playlists = get_playlists(user_id)
    print("Found "+str(len(playlists))+" playlists")

    for playlist_id in playlists:
        playlist_name = sp.playlist(playlist_id)['name']
        print(playlist_id)
        track_ids,artist_ids = get_playlist_tracks(playlist_id)
        print(len(track_ids),len(artist_ids))
        if len(track_ids)==0:
            continue
        for i,track_id in enumerate(track_ids):
            track_name = sp.track(track_id)['name']
            artist_id = artist_ids[i]
            print(f"Fetched..{i} ids")

            genres = get_genre_for_track(artist_id)
            if len(genres)==0:
                continue
            new_users_info.loc[i] = [user_id,user_name,playlist_id,playlist_name,track_id,track_name,artist_ids[i],genres]
            i+=1        
            audio_features = sp.audio_features(track_id)[0]
            #print(audio_features.keys())
            if track_id not in list(tracks_info['track_id']): 
                tracks_info.loc[j] = [track_id,artist_id,genres,audio_features['danceability'],audio_features['energy'],audio_features["key"],audio_features["loudness"],audio_features["mode"],audio_features["speechiness"],audio_features["acousticness"],audio_features["instrumentalness"] , audio_features["liveness"],audio_features["valence"],audio_features["tempo"],audio_features["track_href"],audio_features["duration_ms"],audio_features["time_signature"]]
                j+=1
        print("Completed pulling playlist: "+str(playlist_name))
    print("Completed Data Pull")
    try:
        new_users_info.to_csv("New User Tracks.csv",index=False)
        tracks_info_scaled = tracks_info.copy()
        scaling_needed = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",	"liveness", "valence"]
        for feature in scaling_needed:
            tracks_info_scaled[feature] = (tracks_info_scaled[feature] - tracks_info_scaled[feature].min())/(tracks_info_scaled[feature].max() - tracks_info_scaled[feature].min())
        tracks_info_scaled.to_csv("New User's Track Features.csv",index=False)
        print(f"Created New user and Track files")
    except:
        pass
    return new_users_info,tracks_info



def merge_user_track_data(new_users_info,tracks_info):
    """Merges data frame containing users' tracks and track information dataset
       Drops artist and genres variable duplicates
    Args:
        new_users_info (pd.DataFrame): DataFrame containing all users' public tracks
        tracks_info (pd.DataFrame): DataFrame containing all audio features for all tracks

    Returns:
        pd.DataFrame: Combined dataframe
    """
    master = new_users_info.merge(tracks_info,how="left",on="track_id")
    master.drop(['genres_y','artist_id_y'],axis=1,inplace=True)
    master.rename(columns = {"genres_x":"genres","artist_id_x":"artist_id"},inplace=True)
    return master
 

def get_genres_represenentative(master):
    """For each unique genre, returns a dataframe containing one representative song for each genre
        Uses simple aggregation and performs median and mode of audio features of tracks within genre
        One song having multiple genres will be double counted for both genres 
    Args:
        master (pd.DataFrame): User-info + Track information merged dataframe

    Returns:
        pd.DataFrame: Contains audio features for each unique genre combination
    """
    master_explode = master.explode('genres')
    

    track_info_median_cols = ['danceability', 'energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
    track_info_mode_cols = ['key','mode','time_signature']

    a = master_explode.groupby('genres')[track_info_median_cols].agg('median').reset_index()
    b = master_explode.groupby('genres')[track_info_mode_cols].agg(lambda x:st.mode(x).mode[0]).reset_index()
    c = master_explode.groupby('genres')[['track_id']].agg('count').reset_index()

    genre_representative_audio_features = a.merge(b,how="left",on="genres").merge(c,how="left",on="genres")
    genre_representative_audio_features.rename(columns = {"track_id":"num_tracks_in_genre"},inplace=True)
    try:
        genre_representative_audio_features.to_csv("New User's Genre Representative Songs.csv",index=False)
        print("Created Genre Representative file")
    except:
        pass
    return genre_representative_audio_features



#CALLING 
def main(users):
    new_users_info,tracks_info = get_user_tracks(users)

    master = merge_user_track_data(new_users_info,tracks_info)
    representative_song_feature = get_genres_represenentative(master)
    print("----EXECUTED----")
    return representative_song_feature
