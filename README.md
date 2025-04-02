# ArXiv Research Tool

This project is an AI-powered technical article generator that uses ArXiv papers as its knowledge base. The system consists of two main components:

1. **arxivdatabase.py**: A script to fetch and store ArXiv papers in a vector database
2. **arxivapp.py**: A Streamlit web application that generates technical articles based on user queries

## Features

- Fetches thousands of ArXiv papers across multiple computer science categories
- Stores papers in a Chroma vector database with embeddings
- Provides a user-friendly web interface to generate technical articles
- Retrieves relevant research papers based on the user's topic
- Generates structured academic papers with proper citations

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection (for fetching ArXiv data)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HappyHackingSpace/AI-Research-Writers-Tool.git
cd arxiv-research-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Step 1: Build the Vector Database

Run the database builder script to fetch ArXiv papers and create the vector database:

```bash
python arxivdatabase.py
```

This process may take several hours depending on the number of papers you're fetching (default is 5000).

> **Note**: You can adjust the `num_papers` variable in the script to change the number of papers fetched per category.

### Step 2: Launch the Web Application

After building the database, run the Streamlit application:

```bash
streamlit run arxivapp.py
```

The web interface will open in your browser. Enter a topic in the text field and click "Generate Article" to create a technical article based on relevant ArXiv papers.

## How It Works

### arxivdatabase.py

1. Fetches papers from ArXiv across multiple computer science categories
2. Extracts title, authors, summary, and other metadata
3. Splits the text into chunks suitable for embedding
4. Creates embeddings using OpenAI's text-embedding-3-small model
5. Stores the embedded documents in a Chroma vector database

### arxivapp.py

1. Accepts a topic from the user via the Streamlit interface
2. Searches the vector database for relevant research papers
3. Allows the user to select the number of references to use
4. Generates a structured academic paper using OpenAI's GPT-4o model
5. Displays the generated article with proper formatting

## Configuration

You can modify the following parameters in the scripts:

- `num_papers`: Number of papers to fetch per category (default: 5000)
- `results_article`: Maximum number of results per ArXiv API request (default: 200)
- `batch_size`: Number of documents to add to the vector database in each batch (default: 5000)
- `chunk_size`: Size of text chunks for embedding (default: 2000)
- `chunk_overlap`: Overlap between consecutive chunks (default: 100)
- `model`: LLM model used for generation (default: "gpt-4o")
- `temperature`: Creativity parameter for the LLM (default: 0.4)

## Limitations

- The ArXiv API has rate limits, so the fetching process includes sleep intervals
- Generating articles for very niche topics may result in less relevant content
- The quality of generated articles depends on the availability of relevant papers in the database

## License

This project is licensed under the MIT License - see the LICENSE file for details.
