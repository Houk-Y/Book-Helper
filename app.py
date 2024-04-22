import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset from CSV
@st.cache_data
def load_data():
    return pd.read_csv("main_dataset.csv")


df_books = load_data()

# Transform descriptions into TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_books["name"].fillna(""))


# Recommendation function
def recommend_books(previous_book, genre=None):
    # Filter books by genre if genre is specified
    if genre:
        filtered_books = df_books[df_books["category"] == genre]
    else:
        filtered_books = df_books.copy()  # Consider all books if genre is not specified

    # Transform descriptions into TF-IDF vectors
    tfidf_matrix_filtered = tfidf_vectorizer.fit_transform(
        filtered_books["name"].fillna("")
    )

    # Calculate TF-IDF vector for previous book
    previous_book_tfidf = tfidf_vectorizer.transform([previous_book])

    # Calculate cosine similarity between previous book and filtered books
    similarities = cosine_similarity(
        previous_book_tfidf, tfidf_matrix_filtered
    ).flatten()

    # Sort books based on similarity scores
    filtered_books["similarity"] = similarities
    recommended_books = filtered_books.sort_values(by="similarity", ascending=False)[
        [
            "name",
            "author",
            "format",
            "book_depository_stars",
            "price",
            "currency",
            "old_price",
            "isbn",
            "category",
            "img_paths",
        ]
    ]

    return recommended_books


# Streamlit UI
st.title("Book Recommendation System")

# User input
previous_book = st.text_input("What was your previous favorite book?")
genre = st.selectbox(
    "What genre of books do you like to read? (Optional)",
    [""] + list(df_books["category"].unique()),  # Include an empty option
)

if st.button("Get Recommendations"):
    if previous_book.strip() != "":
        # Get recommendations
        recommended_books = recommend_books(previous_book, genre)

        # Display recommended books
        if not recommended_books.empty:
            st.subheader("Recommended books:")
            st.dataframe(recommended_books)
        else:
            st.write("Sorry, no recommendations found based on your input.")
    else:
        st.write("Please enter your previous favorite book to get recommendations.")
