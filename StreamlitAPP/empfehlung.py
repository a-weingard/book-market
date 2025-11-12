# empfehlung.py

import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import torch
import umap
from sentence_transformers import SentenceTransformer, util
from wordcloud import WordCloud
from io import BytesIO
import matplotlib.pyplot as plt

# This module is used to display book recommendations based on two main options:
# content-based recommendations or filter-based recommendations.

# DATA AND MODEL LOADING

# Load the book dataset
@st.cache_data
def load_data():
    """
    Loads the book dataset from a CSV file and converts the genre column 
    from a string representation to actual Python lists.

    Returns:
        pd.DataFrame: The processed book dataset with parsed genre lists.
    """
    books = pd.read_csv("data/final_books_recommend.csv")

    # Convert the "genre_list" column from string to actual list
    books['genre_list'] = books['genre_list'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )

    return books

# Load the pre-trained "sentence transformer" model
@st.cache_resource
def load_model():
    """
    Loads a pre-trained SentenceTransformer model to generate text embeddings.

    Returns:
        SentenceTransformer: The loaded language model.
    """
    return SentenceTransformer('all-mpnet-base-v2')

# Compute text embeddings for all original book descriptions
@st.cache_data
def compute_embeddings(_model, descriptions):
    """
    Computes text embeddings for a list of book descriptions using the loaded model.

    Parameters:
        _model: The pre-trained language model (SentenceTransformer).
        descriptions: A list of book descriptions as strings ("descriptions" column).

    Returns:
        tensor: The generated embeddings as a 2D-tensor (PyTorch-Tensor).
    """
    return _model.encode(descriptions, convert_to_tensor=True)

# load data and model before start
books = load_data()
model = load_model()
descriptions = books['description'].tolist()
embeddings = compute_embeddings(model, descriptions)

# Function for similarity-based recommendation
def find_similar_books(title, df, model, embeddings, top_n=5):
    """
    Finds books similar to a selected title based on the description using 
    cosine similarity between the target embedding und all other embeddings.

    Parameters:
        title (str): The title of the book to compare.
        df (pd.DataFrame): The complete book dataset.
        model (SentenceTransformer): The model to generate text embeddings.
        embeddings (tensor): Computed embeddings of all book descriptions ("decription" column od a df).
        top_n (int): Number of similar books to return (default=5).

    Returns:
        list: A list of tuples of similar books (index, similarity_score, title), ranked by cosine similarity.
    """
    # Find the row of the selected book 
    target_row = df[df['title'] == title]

    # Return an empty list if the row is empty
    if target_row.empty:
        return []

    # Get the description of the target book and compute its embedding
    target_desc = target_row.iloc[0]['description']
    target_embedding = model.encode(target_desc, convert_to_tensor=True)

    # Calculate cosine similarity between the target and all other books in the dataset
    cos_sim = util.cos_sim(target_embedding, embeddings)[0]
    top_results = torch.topk(cos_sim, k=top_n + 1)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        score = score.item()
        if df.iloc[idx]['title'] != title: 
            results.append((idx, score, df.iloc[idx]['title']))
        if len(results) == top_n:
            break

    return results

# Function for word cloud generation
def generate_wordcloud(text):
    """
    Generates a word cloud image from the given text and returns it as an in-memory image buffer.

    Parameters:
        text (str): The input text used to generate the word cloud (from the "clean_description" column, i.e description without stopwords) .

    Returns:
        BytesIO: A buffer containing the PNG image of the generated word cloud.
    """
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        random_state=42
    ).generate(text)

    # Store generated image in memory buffer
    img_buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear') # glättet die Pixel
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    return img_buffer

# Function for book cover image
def get_book_cover(isbn):
    """
    Retrieves a book cover image using the ISBN via the Open Library API.
    If no valid image is found, returns a placeholder image URL.

    Parameters:
        isbn (str): The ISBN number of the book ("isbn10" column in the df).

    Returns:
        str: The URL of the book cover or a fallback placeholder image.
    """
    # If ISBN is missing, return a placeholder image
    if pd.isna(isbn) or isbn == '':
        return "https://via.placeholder.com/120x180.png?text=No+Cover"

    cover_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
    
    # Check if the image exists at the URL
    response = requests.get(cover_url)
    if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image"):
        return cover_url
    else:
        # If image is not found, display fallback placeholder
        return "https://placehold.co/120x180?text=No+Cover&font=roboto"

# STREAMLIT UI 
def show():
    # HTML code for title and subtitle styling
    st.markdown(
        """
        <style>
        .title {
            font-size:60px;
            font-weight:bold;
            color:#FFD700;
            text-align:center;
            padding-top:30px;
        }
        .subtitle {
            font-size:32px;
            color:#bbbbbb;
            text-align:center;
            padding-bottom:20px;
        }
        .info-box {
            background-color: #111827;
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px;
        }
        .section-title {
            font-size: 22px;
            color: #4ade80;
            margin-bottom: 10px;
        }
        .content-text {
            font-size: 18px;
            line-height: 1.6;
            color: black;
        }
        </style>
        <div class='title'>📚 Bücher entdecken nach deinem Geschmack</div>
        <div class='subtitle'> Empfehlungen basierend auf Inhalt und Stil </div>
        """,
        unsafe_allow_html=True,
    )

# Recommendation option selection
    st.markdown("### Wähle aus, wie du Buchempfehlungen erhalten möchtest:")
    option = st.radio(
        label=" ",
        options=("Finde ähnliche Bücher zu deinem Favoriten", "Finde Bücher nach Genre, Bewertung und mehr"),
        key="empfehlung_option_radio_main"
    )

# Filter selection (for both options)
    st.markdown("#### Du hast folgende Filteroptionen (optional)")
    with st.expander("Filter anzeigen / ausblenden"):

        min_rating = st.slider(
            "Minimale durchschnittliche Bewertung",
            min_value=0.0,
            max_value=5.0,
            #value=3.5,
            step=0.1,
            help="Wie viel sollte die minimale Bewertung sein?",
            key="min_avg_rating"
        )

        selected_year = st.slider(
            "Erscheinungsjahr",
            min_value=1811,
            max_value=2018,
            value=(1811, 2018),
            step=1,
            help="Begrenzt die Auswahl auf Bücher aus bestimmten Jahren.",
            key="filter_years"
        )

        selected_genres = st.multiselect(
            "Genres (mehrere möglich)",
            options=sorted({genre for sublist in books['genre_list'].dropna().tolist() for genre in sublist}),
            key="filter_genres"
        )
        
        # Initialize author filter (only for the 2nd option "Finde Bücher nach Genre, Bewertung und mehr")
        selected_author = None

        if option == "Über Filter (Genre, Bewertung etc.)":
            selected_author = st.selectbox(
                "Autor (optional)",
                options=["Alle Autoren"] + sorted(books['author'].unique().tolist()),
                index=0,
                key="filter_author"
            )

        num_results = st.selectbox(
            "Wie viele Bücher möchtest du angezeigt bekommen?",
            options=[5, 10, 20],
            index=0,
            key="select_num_results"
        )

# Option 1: similar books based on selected title
    if option == "Finde ähnliche Bücher zu deinem Favoriten":
        selected_title = st.selectbox(
            "Wähle ein Buch aus der Liste aus, das dir gefallen hat", 
            sorted(books['title'].unique()),
            index=None,
            placeholder="Buchtitel eingeben oder auswählen...",
            key="book_selectbox")

        if not selected_title:
            st.info(f"Bitte wähle ein Buch aus der Liste.")
            return
        
        recommendations = []

        show_similar_books = st.button("Ähnliche Bücher finden")

        # Display selected book details 
        selected_book = books[books['title'] == selected_title].iloc[0]
        year = selected_book['publication_year']
        book_author = selected_book['author']
        rating = selected_book['avg_rating']
        count_ratings = selected_book['ratings_count']
        book_genres = selected_book['genres']
        isbn = selected_book['isbn']
        cover_url = get_book_cover(isbn)
        book_description = selected_book['description']

        st.markdown("### Dein gewähltes Buch")
            
        with st.container():
            left_col, middle_col, right_col = st.columns([1, 2, 2])

            with left_col:
                st.image(cover_url, caption=f"ISBN: {isbn}", width=200)

            with middle_col:
                st.markdown(f"### {selected_title} ({year})")
                st.markdown(f"**Autor:** {book_author}")
                st.markdown(f"**Genres:** {book_genres}")
                st.markdown(f"**Bewertung:** {rating} ({count_ratings} Bewertungen)")

                with st.expander("Buchbeschreibung anzeigen"):
                    st.write(f"{book_description}")

            with right_col:
                wc_img = generate_wordcloud(selected_book['clean_description'])
                st.image(wc_img, use_container_width=True) #, caption="Wordcloud zur Beschreibung"

        st.markdown("---")

        recommendations = []

        if show_similar_books:

            st.markdown("### Ähnliche Bücher")

            recommendations = find_similar_books(
                title=selected_title,
                df=books,
                model=model,
                embeddings=embeddings,
                top_n=50 # return many books so the filter can be applied to them
            )

            filtered_recommendations = []
            for idx, score, title in recommendations:
                row = books.iloc[idx]

                # Apply filters
                if (
                    row['avg_rating'] >= min_rating and
                    selected_year[0] <= row['publication_year'] <= selected_year[1] and
                    all(genre in row['genre_list'] for genre in selected_genres)
                ):
                    filtered_recommendations.append((row, score))

            # If there are no books matching the filters
            if not filtered_recommendations:
                st.warning("Keine passenden Buchempfehlungen gefunden. Bitte passe deine Filter an.")
            # Display books matching the filters
            else:
                for book_row, score in filtered_recommendations[:num_results]:
                    with st.container():
                        left_col, middle_col, right_col = st.columns([1, 2, 2])
                        with left_col:
                            st.image(get_book_cover(book_row['isbn']), width=150)
                        with middle_col:
                            st.markdown(f"### {book_row['title']} ({book_row['publication_year']})")
                            st.markdown(f"**Autor:** {book_row['author']}")
                            st.markdown(f"**Genres:** {book_row['genres']}")
                            st.markdown(f"**Bewertung:** {book_row['avg_rating']} ({book_row['ratings_count']} Bewertungen)")
                            st.markdown(f"**Ähnlichkeit:** {100 * score:.1f}%")

                            with st.expander("Buchbeschreibung anzeigen"):
                                st.write(book_row['description'])

                        with right_col:
                            wc_img = generate_wordcloud(book_row['clean_description'])
                            st.image(wc_img, use_container_width=True)

                    st.markdown("---") 

# Option 2: Filter-only option
    elif option == "Finde Bücher nach Genre, Bewertung und mehr":
        selected_author = st.selectbox(
            "Autor (optional)",
            options=["Alle Autoren"] + sorted(books['author'].dropna().unique().tolist()),
            index=0,
            key="filter_author"
        )

        show_filtered_books = st.button("Bücher finden")

        filtered_df = books.copy()

        # Apply author filter
        if selected_author != "Alle Autoren":
            filtered_df = filtered_df[filtered_df['author'] == selected_author]

        # Apply avg_ryting and publication_year filters
        filtered_df = filtered_df[
            (filtered_df['avg_rating'] >= min_rating) &
            (filtered_df['publication_year'] >= selected_year[0]) &
            (filtered_df['publication_year'] <= selected_year[1])
        ]

        # Apply genre filter
        if selected_genres:
            filtered_df = filtered_df[
                filtered_df['genre_list'].apply(
                    lambda genre_list: any(selected in genre_list for selected in selected_genres)
                )
            ]

        filtered_df = filtered_df.sort_values(by="avg_rating", ascending=False)

        # Display the number of books that match the filters
        if show_filtered_books:
            st.markdown(f"**{len(filtered_df)} Bücher gefunden**")

            # If there are no books matching the filters
            if filtered_df.empty:
                st.warning("Leider keine Bücher gefunden. Bitte passe deine Filter an.")

            # Display books matching the filters (image and book details)
            else:
                for _, row in filtered_df.head(num_results).iterrows():
                    with st.container():
                        left_col, right_col = st.columns([1, 4])
                        with left_col:
                            st.image(get_book_cover(row['isbn']), width=200)
                        with right_col:
                            st.markdown(f"### {row['title']} ({row['publication_year']})")
                            st.markdown(f"**Autor:** {row['author']}")
                            st.markdown(f"**Genres:** {row['genres']}")
                            st.markdown(f"**Bewertung:** {row['avg_rating']}")
                    st.markdown("---")

 # Run the app   
show()
