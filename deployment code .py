import pandas as pd
import numpy as np

import streamlit as st
import pickle

import nltk
import sklearn

from sklearn.decomposition import TruncatedSVD

data = pd.read_csv('devo_merge_new.csv')
columns = ['Album_Name', 'Track_Name', 'Popularity']
new_df = data[['Album_Name', 'Track_Name', 'Popularity']]

ls_crosstab = new_df.pivot_table(values='Popularity', index='Album_Name', columns='Track_Name', fill_value=0)
X = ls_crosstab.T
SVD = TruncatedSVD(n_components=3, random_state=5)
resultant_matrix = SVD.fit_transform(X)
corr_matrix = np.corrcoef(resultant_matrix)

def get_recommendations(song_name):
    col_idx = ls_crosstab.columns.get_loc(song_name)
    corr_specific = corr_matrix[col_idx]
    sim_scores = list(enumerate(corr_specific))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    indices = [i[0] for i in sim_scores]
    return data['Track_Name'].iloc[indices]

    
    
st.title("Recommendation System For Devotional Songs")
 
song_list = new_df['Track_Name'].values
selected_song = st.selectbox("Type or select a song from the dropdown",song_list)

##sel = st.text_input(selected_song)
if st.button('Search'):
    st.write(selected_song)
    recommended_song_names = get_recommendations(selected_song)
    st.subheader("Recommended songs For You")
    for i in recommended_song_names:
        st.write(i)


