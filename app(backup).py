import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from IPython.display import IFrame
import pickle
import time

def main():
    st.title(':dog2: Parkies Recommendations :dog2:')
    st.sidebar.image('parkie.png', use_column_width="always")
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
    st.sidebar.header("Search for your favorite song")
    
    track_search = st.sidebar.text_input("Enter the title of your song", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
    artist_search = st.sidebar.text_input("Enter the name of the artist (optional)", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")

    if artist_search and track_search:
        query = f'artist:{artist_search} track:{track_search}'
    elif track_search:
        query = track_search
    else:
        print("Error: Please provide at least a song name.")
        return []
    results = sp.search(q=query, limit=3)
    
    st.subheader("Choose the right song from the list below")

    for i, track in enumerate(results['tracks']['items']):
            artists = ', '.join(artist['name'] for artist in track['artists'])
            song_name = track['name']
            track_id = track['id']
            chosen_track_id=""
            # st.write(f"{artists} - {song_name}")
            st.write(f'<iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator" width="700" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)

            if st.button(f'Select this song', key=f'select_button_{i+1}'):
                chosen_track_id = track_id
                break
    else:
            chosen_track_id = None

    if chosen_track_id:
        # st.text(f"Selected Track ID: {chosen_track_id}")

      
        song=sp.audio_features(chosen_track_id)

        song_expl=sp.track(chosen_track_id)['explicit']
        song_pop = sp.track(chosen_track_id)['popularity']
        
        song_df= pd.DataFrame(song)
        song_df['is_explicit']=song_expl
        song_df['popularity']=song_pop
        selected_columns = ['is_explicit','popularity','danceability', 'energy', 'key', 
                        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'tempo', 'time_signature']
        song_df = song_df[selected_columns]

        #choose the right model 
        
    
        #if function depending on each song


        #Import pickles
        with open('km100.pickle', 'rb') as handle:
            km100 = pickle.load(handle)

        with open('scaler.pickle', 'rb') as handle:
            scaler = pickle.load(handle)


        with st.spinner('Wait for it... :dog2: Parkie is thinking :dog2:'):
            time.sleep(3)


        scaled_song = scaler.transform(song_df)
        prediction=km100.predict(scaled_song)

        st.balloons()
        # st.text(f"Recommended cluster:{prediction[0]}")
        
        clusters_df= pd.read_csv("tracks_clustered_df.csv")

        random_list = []

        
        random_value = clusters_df['track_id'].sample(n=1).iloc[0]
        random_list.append(random_value)
        random_value2 = clusters_df['track_id'].sample(n=1).iloc[0]
        random_list.append(random_value2)
        random_value3 = clusters_df['track_id'].sample(n=1).iloc[0]
        random_list.append(random_value3)

        # clusters_df[clusters_df['cluster_km100']== prediction[0]]
        st.subheader("Your recommendations:")
        st.write(f'<iframe src="https://open.spotify.com/embed/track/{random_list[0]}?utm_source=generator" width="320" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)
        st.write(f'<iframe src="https://open.spotify.com/embed/track/{random_list[1]}?utm_source=generator" width="320" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)
        st.write(f'<iframe src="https://open.spotify.com/embed/track/{random_list[2]}?utm_source=generator" width="320" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
