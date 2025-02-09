import pandas as pd
import numpy as np
import torch
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr



books = pd.read_csv(r"dataset\books_with_emotions.csv")

# create a URL for a larger version of the thumbnail image
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# replaces missing books thumbnail by book_not_found.jpg
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    r"book_not_found.jpg",
    books["large_thumbnail"]
)



# Determine if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize HuggingFaceEmbeddings with the desired model
model_name = "all-MiniLM-L6-v2"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load the existing database
persist_dir = r"chroma_db"

db_books = Chroma(
    persist_directory=persist_dir,
    embedding_function=hf_embeddings
)




def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    """
    Retrieves semantic recommendations for books based on a query, category, and tone.

    Args:
        query (str): The search query to find relevant books.
        category (str, optional): The category to filter the results by. Defaults to None, which means no category filtering.
        tone (str, optional): The desired tone of the books.  Can be "Happy", "Surprising", "Angry", "Suspenseful", or "Sad". Defaults to None, which means no tone-based sorting.
        initial_top_k (int, optional): The number of initial results to retrieve from the similarity search. Defaults to 50.
        final_top_k (int, optional): The number of top recommendations to return. Defaults to 16.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended books, sorted and filtered according to the provided criteria.  The DataFrame is likely to contain information about each book, such as ISBN, title, category, and tone scores.
    """

    # Semantic similarity search
    recs = db_books.similarity_search(query, k=initial_top_k)
    # Extract ISBNs from search results.
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    # Filter books DataFrame by ISBNs.
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Filter by category
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort by tone
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs




def recommend_books(
    query: str,
    category: str,
    tone: str):

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results



categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    # Header Section
    gr.Markdown("""
    # 📚 Semantic Book Recommender
    *Semantic search powered by LLM*
    """)

    # Input Section
    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="Describe your ideal book",
                placeholder="e.g., 'A coming-of-age story set in 1980s Tokyo'",
                lines=2,
                max_lines=4
            )
            examples = gr.Examples(
                examples=[
                    ["A philosophical sci-fi about artificial consciousness"],
                    ["Cozy mystery in a small English village"],
                    ["Epic fantasy with complex political intrigue"]
                ],
                inputs=[user_query]
            )

        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories,
                value="All",
                label="Book Category",
                interactive=True
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                value="All",
                label="Desired Mood",
                interactive=True
            )
            with gr.Row():
                submit_btn = gr.Button("Search", variant="primary")
                clear_btn = gr.Button("Clear")

    # Results Section
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)



    # Documentation Section
    with gr.Accordion("How it works", open=False):
        gr.Markdown("""
        - **Semantic Search**: Matches meaning beyond just keywords
        - **AI Recommendations**: Based on book descriptions and themes
        - **Filters**: Narrow by category and emotional tone
        """)

    # Event Handling
    submit_btn.click(
        fn=lambda: gr.Info("Searching our library..."),
        outputs=None,
        queue=False
    ).then(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
        api_name="recommend"
    )

    clear_btn.click(
        lambda: [None, None, None, None],
        outputs=[user_query, category_dropdown, tone_dropdown, output],
        queue=False
    )




if __name__ == "__main__":

  dashboard.launch()