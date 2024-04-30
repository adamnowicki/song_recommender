import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import times
import sklearn
import time

import os
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


def main():
    st.title(':dog2: Parkies Recommendations :dog2:')
    st.sidebar.image('images/parkie.png', use_column_width="always")
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET))
    st.sidebar.header("Search for your favorite song")
    
    track_search = st.sidebar.text_input("Enter the title of your song", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
    artist_search = st.sidebar.text_input("Enter the name of the artist (optional)", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
    st.sidebar.write("The project was created during Ironhack Data Analytics Bootcamp Paris (2024) by: Adam Nowicki, Javier Peyriere, Martino Ossandon Busch and Smita Prakas")
    
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
            st.write(f'<iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator" width="100" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)

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
        list_simple = ['danceability', 'energy','speechiness', 'acousticness', 'instrumentalness','valence']

        def weight_on(song, list_simple):
            min_a = 1
            min_b = 1
            col_a = 'danceability'
            col_b = 'danceability'
            df_simple = song[[f for f in list_simple]]
            for f in df_simple.columns:
                if df_simple[f][0] < min_a:
                    min_a = df_simple[f][0]
                    col_a = f
            for f in df_simple.columns:
                if (1-df_simple[f][0]) < min_b:
                    min_b = 1-df_simple[f][0]
                    col_b = f
            if min_a < min_b:
                return col_a
            else:
                return col_b
        
        feat = weight_on(song_df, list_simple)
        
        audio_features_model = [ 'danceability', 'energy',
       'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'valence', 'tempo']

        if feat == 'danceability':

            song_w_df = song_df[[f for f in audio_features_model]]
            song_w_df['weight_1'] = song_w_df[feat] # Changed to only one weight, was too much!
            
            with open('pickles/model_km_danc.pickle', 'rb') as handle:
                km_danc = pickle.load(handle)

            with open('pickles/scaler_danc.pickle', 'rb') as handle:
                scaler_danc = pickle.load(handle)
            
            scaled_song = scaler_danc.transform(song_w_df)
            prediction=km_danc.predict(scaled_song)

            clusters_df= pd.read_csv("data/tracks_clustered_df_da.csv")
            df_good=clusters_df[clusters_df['cluster_km_danc']==prediction[0]]
            suggestion=df_good['track_id'].sample(n=1).iloc[0]
            suggestion2=df_good['track_id'].sample(n=1).iloc[0]
            
        # call pickle for each & run test
            
        elif feat == 'energy':
            
            song_w_df= song_df[[f for f in audio_features_model]]
            song_w_df['weight_1'] = song_w_df[feat]

            with open('pickles/model_km_ener.pickle', 'rb') as handle:
                    km_ener = pickle.load(handle)

            with open('pickles/scaler_ener.pickle', 'rb') as handle:
                    scaler_ener = pickle.load(handle)

            scaled_song_en = scaler_ener.transform(song_w_df)
            prediction = km_ener.predict(scaled_song_en)
            


            clusters_df= pd.read_csv("data/tracks_clustered_df_en.csv")
            df_good=clusters_df[clusters_df['cluster_km_ener']==prediction[0]]
            suggestion=df_good['track_id'].sample(n=1).iloc[0]
            suggestion2=df_good['track_id'].sample(n=1).iloc[0]

        # call pickle for each & run test
            
        elif feat == 'speechiness':
            song_w_df = song_df[[f for f in audio_features_model]]
            song_w_df['weight_1'] = song_w_df[feat]

            with open('pickles/model_km_sp.pickle', 'rb') as handle:
                km_sp = pickle.load(handle)

            with open('pickles/scaler_sp.pickle', 'rb') as handle:
                scaler_sp = pickle.load(handle)

            scaled_song_sp = scaler_sp.transform(song_w_df)
            prediction = km_sp.predict(scaled_song_sp)


            clusters_df= pd.read_csv("data/tracks_clustered_df_sp.csv")
            df_good=clusters_df[clusters_df['cluster_km_sp']==prediction[0]]
            suggestion=df_good['track_id'].sample(n=1).iloc[0]
            suggestion2=df_good['track_id'].sample(n=1).iloc[0]

        # call pickle for each & run test
        elif feat == 'acousticness':
            song_w_df = song_df[[f for f in audio_features_model]]
            song_w_df['weight_1'] = song_w_df[feat]

            with open('pickles/model_km_ac.pickle', 'rb') as handle:
                km_ac = pickle.load(handle)

            with open('pickles/scaler_ac.pickle', 'rb') as handle:
                scaler_ac = pickle.load(handle)

            scaled_song_ac = scaler_ac.transform(song_w_df)
            prediction = km_ac.predict(scaled_song_ac)

            clusters_df= pd.read_csv("data/tracks_clustered_df_ac.csv")
            df_good=clusters_df[clusters_df['cluster_km_ac']==prediction[0]] 
            suggestion=df_good['track_id'].sample(n=1).iloc[0]
            suggestion2=df_good['track_id'].sample(n=1).iloc[0]

        

        # call pickle for each & run test
        elif feat == 'instrumentalness':
            song_w_df = song_df[[f for f in audio_features_model]]
            song_w_df['weight_1'] = song_w_df[feat]

            with open('pickles/model_km_ins.pickle', 'rb') as handle:
                km_ins = pickle.load(handle)
            with open('pickles/scaler_ins.pickle', 'rb') as handle:
                scaler_ins = pickle.load(handle)
            
            scaled_song_ins = scaler_ins.transform(song_w_df)
            prediction = km_ins.predict(scaled_song_ins)


            clusters_df= pd.read_csv("data/tracks_clustered_df_ins.csv")
            df_good=clusters_df[clusters_df['cluster_km_ins']==prediction[0]] 
            suggestion=df_good['track_id'].sample(n=1).iloc[0]
            suggestion2=df_good['track_id'].sample(n=1).iloc[0]
            
            

        # call pickle for each & run test
        elif feat == 'valence':
            song_w_df = song_df[[f for f in audio_features_model]]
            song_w_df['weight_1'] = song_w_df[feat]

            with open('pickles/model_km_val.pickle', 'rb') as handle:
                km_val = pickle.load(handle)

            with open('pickles/scaler_val.pickle', 'rb') as handle:
                scaler_val = pickle.load(handle)

            scaled_song_val = scaler_val.transform(song_w_df)
            prediction = km_val.predict(scaled_song_val)

            clusters_df= pd.read_csv("data/tracks_clustered_df_val.csv")
            df_good=clusters_df[clusters_df['cluster_km_val']==prediction[0]]
            suggestion=df_good['track_id'].sample(n=1).iloc[0]
            suggestion2=df_good['track_id'].sample(n=1).iloc[0]
            
        else:
            pass
    

        with st.spinner('Wait for it... :dog2: Parkie is thinking :dog2:'):
            time.sleep(3)

        st.balloons()
        # st.text(f"Recommended cluster:{prediction[0]}")
        

        # clusters_df[clusters_df['cluster_km100']== prediction[0]]
        st.subheader("Your recommendations:")
        st.write(f'<iframe src="https://open.spotify.com/embed/track/{suggestion}?utm_source=generator" width="700" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)
        st.write(f'<iframe src="https://open.spotify.com/embed/track/{suggestion2}?utm_source=generator" width="700" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
