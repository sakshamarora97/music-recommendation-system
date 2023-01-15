
DESCRIPTION:
+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| File Type     | File Name             | Description                                                                               |
+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| Frontend File | EXPLORE.twb           | User-Facing Tableau Dashboard with "Spotify User ID" input field for user                 |
+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| Python Script | recommendation_v0.py  | Main Backend Script triggered on entering "Spotify User ID" via TabPy                     |
|               |                       | Input: Spotify User ID                                                                    |
|               |                       | Output:                                                                                   |
|               |                       | df_recommendation_overall.csv (Old + Top22 + Top Overall)                                 |
|               |                       | user-user_v2.csv                                                                          |
|               |                       | Dependencies: get_new_user_songs.py, rating_prep.py                                       |
+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| Python Script | get_new_user_songs.py | Secondary script to fetch user's public playlist and track features                       |
|               |                       | Input: Spotify User ID                                                                    |
|               |                       | Output:                                                                                   |
|               |                       | New User Tracks.csv                                                                       |
|               |                       | New User's Genre Representative Songs.csv                                                 |
|               |                       | New User's Track Features.csv                                                             |
+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| Python Script | rating_prep.py        | Secondary script to transform the raw listening history into user-song interaction matrix |
|               |                       | with ratings based on calculated tf-idf scores                                            |
|               |                       | Input: userid-timestamp-artid-artname-traid-traname.tsv                                   |
|               |                       | Output: user_song_interacting.csv                                                         |
+---------------+-----------------------+-------------------------------------------------------------------------------------------+
| Python Script | evaluation.py         | Script to test performance of recommendation system based on both absolute                |
|               |                       | predicted values and relative ranking(relevance)                                          |
|               |                       | Dependencies: recommendation_v0.py                                                        |
+---------------+-----------------------+-------------------------------------------------------------------------------------------+
|               |                       |                                                                                           |
+---------------+-----------------------+-------------------------------------------------------------------------------------------+

INSTALLATION
Assumes installation of Tableau, Python and Anaconda Base Package
Step 1. Install TabPy (Tutorial https://www.youtube.com/watch?v=Xk67f45BuoA)
Step 2. Check directory where python packages are installed in your system e.g. Enter: "pip show numpy" in terminal 
Step 3. Place all downloaded files in the same directory

EXECUTION
Step 1. Find your spotify ID: Go to https://open.spotify.com/, Navigate to Profile, and from the address bar extract alphanumeric string.
E.g. If your link is "https://open.spotify.com/user/31isrsytv6r4x6tg3biw3mfaax2i", your user id is : 31isrsytv6r4x6tg3biw3mfaax2i
Step 2. Launch TabPy with command "tabpy" in terminal
Step 3. Open Explore.twb in Tableau and connect to TabPy
Step 4. Open Explore.twb and Navigate to Recommendation Dashbaord. 
Enter Spotify User ID and fly away!
