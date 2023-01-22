# we are going to create a streamlit app to recommend the airbnb listings based on the user's id inputs

# import the necessary libraries

import streamlit as st # for creating the web app
import pandas as pd # for data manipulation
import numpy as np # for data manipulation
from sklearn.metrics.pairwise import cosine_similarity # for calculating the cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer # for creating the tf-idf matrix

# load the data
@st.cache
def load_data():
    listings = pd.read_csv('listings.csv')
    return listings

# create the tf-idf matrix
@st.cache
def create_tfidf_matrix(listings):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(listings['description'])
    return tfidf_matrix

# calculate the cosine similarity
@st.cache
def calculate_cosine_similarity(tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# recommend the listings
def recommend_listings(cosine_sim, listings, id, top_n): # id is the id of the listing
    recommended_listings = [] # create an empty list to store the recommended listings
    idx = id - 1 # get the index of the listing
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False) # get the similarity scores in descending order
    top_n_indexes = list(score_series.iloc[1:top_n+1].index) # get the indexes of the top n similar listings
    for i in top_n_indexes: # append the names of the recommended listings to the list
        recommended_listings.append(list(listings['name'])[i]) # append the names of the recommended listings to the list
    return recommended_listings # return the list of recommended listings

# create the web app

# set the page config
st.set_page_config(page_title='Airbnb Seattle Listings Recommender', page_icon=':house:', layout='wide', initial_sidebar_state='auto')
    
# set the title of the web app
st.title('Airbnb Seattle Listings Recommender')

# load the data
listings = load_data()

# show the 5 rows of the data
st.write(listings.head())

# create the tf-idf matrix
tfidf_matrix = create_tfidf_matrix(listings)

# calculate the cosine similarity
cosine_sim = calculate_cosine_similarity(tfidf_matrix)

# create the sidebar
st.sidebar.header('User Input Features')
# create the user input features
id = st.sidebar.number_input('Enter the id of the listing', min_value=1, value=1)

# create the recommendation button
if st.sidebar.button('Recommend'):
    recommended_listings = recommend_listings(cosine_sim, listings, id, 10)
    st.header('Recommended Listings')
    for i in recommended_listings:
        st.write(i)

# write the name of the listing
st.header('Listing Name')

st.success(listings['name'][id-1])

# write the price of the listing

st.header('Listing Price')

st.success(listings['price'][id-1])



# create a subheader
st.subheader('Listing Description')

# show the description of the listing
st.write(listings['description'][id-1])

# show the image of the listing
st.image(listings['picture_url'][id-1])



