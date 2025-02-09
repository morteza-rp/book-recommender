# 📚 Semantic Book Recommender with LLM

Welcome to the **Semantic Book Recommender with LLM**! This project leverages the power of semantic search and natural language processing to help you discover books that match your interests, mood, and preferences. Whether you're looking for a philosophical sci-fi novel or a cozy mystery, this tool will provide personalized recommendations based on your input.
<video src="dataset\2025-02-09-23-36-40.mp4" width="1000" height="800" controls></video>
---

## 🚀 Features

- **Semantic Search**: Find books based on the meaning of your query, not just keywords.
- **AI-Powered Recommendations**: Get book suggestions based on descriptions, themes, and emotional tones.
- **Custom Filters**: Narrow down recommendations by category and mood.
- **Interactive Interface**: Built with Gradio for an intuitive and user-friendly experience.
- **GPU Support**: Utilizes CUDA for faster processing if a GPU is available.

---

## 🛠️ Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/morteza-rp/book-recommender-with-LLM.git
   cd book-recommender-with-LLM
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Ensure the dataset `books_with_emotions.csv` is placed in the `dataset` folder.
   - Place the `chroma_db` directory (containing the precomputed embeddings) in the project root.

4. **Run the application**:
   ```bash
   python gradio_dashboard.py
   ```

---

## 🎯 How It Works

1. **Input Your Query**:
   - Describe your ideal book in the textbox (e.g., "A philosophical sci-fi about artificial consciousness").
   - Optionally, select a category (e.g., "Science Fiction") and a mood (e.g., "Suspenseful").

2. **Get Recommendations**:
   - The system uses semantic search to find books that match your query.
   - Results are filtered and sorted based on the selected category and mood.

3. **Explore Recommendations**:
   - View book thumbnails, titles, authors, and truncated descriptions.
   - Click on a book to learn more or purchase.

---

## 🧠 Technical Details

- **Embeddings**: The project uses Hugging Face's `all-MiniLM-L6-v2` model for generating text embeddings.
- **Vector Database**: Chroma is used to store and retrieve book embeddings efficiently.
- **Gradio Interface**: The user interface is built using Gradio, making it easy to interact with the recommendation system.
- **GPU Acceleration**: The system automatically detects and utilizes CUDA if a GPU is available.

---

## 📂 Project Structure

```
Semantic Book Recommender with LLM/
├── dataset/
│   └── books.csv
|   └── books_cleaned.csv
|   └── books_with_categories.csv
|   └── books_with_emotions.csv
|
├── chroma_db/
│   └── (precomputed embeddings)
|
├── gradio_dashboard.py
├── requirements.txt
└── README.md
```

---

## 🤖 Example Queries

- "A philosophical sci-fi about artificial consciousness"
- "Cozy mystery in a small English village"
- "Epic fantasy with complex political intrigue"

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the embedding model.
- [LangChain](https://www.langchain.com/) for the semantic search framework.
- [Gradio](https://www.gradio.app/) for the interactive interface.

---

Enjoy discovering your next favorite book! 📖✨
