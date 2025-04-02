import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model="gpt-4o", temperature=0.4)

vector_db = Chroma(
    collection_name="arxiv_collection",
    persist_directory="./chroma_arxiv_db",
    embedding_function=embeddings,
)

st.title("📄 AI-Powered Technical Article Generator")

topic = st.text_input("📝 Enter a topic for the article:")

def search_research_papers(query, top_k=10):
    """Search for the most relevant papers in the vector database related to the given topic and get metadata."""
    docs = vector_db.similarity_search_with_relevance_scores(query, k=top_k)

    formatted_references = []
    for i, (doc, score) in enumerate(docs):
        title = doc.metadata.get(
            "title", f"Unknown Title {i+1}"
        ) 
        link = doc.metadata.get("link", "#")  
        formatted_references.append(f"[{i+1}] {title}. Available: {link}")

    return formatted_references


def generate_prompt(topic, references):
    """Create a prompt for generating an academic paper."""
    references_text = "\n".join(references)  

    return f"""
    Generate a **structured and well-organized academic research paper** on the topic: **{topic}**  
    following the conventions of theoretical computer science research.

    ## **Paper Structure:**
    1. **Abstract**: Summarize the research scope, key findings, and relevance.
    2. **Introduction**: Define the research problem, motivation, and significance.
    3. **Background & Literature Review**: Present key studies, theoretical foundations, and recent advancements.
    4. **Research Problem & Objectives**: Clearly outline research questions and goals.
    5. **Theoretical Framework**: Explain relevant models, algorithms, or formal methods.
    6. **Analysis & Discussion**: Evaluate existing methods, limitations, and possible improvements.
    7. **Conclusion**: Summarize findings, research limitations, and future directions.
    8. **References**: Provide IEEE-formatted citations at the end. Sort the references from first used to last used
    
    ## **Relevant Scientific Paper References:**
    {references_text}
    """

def generate_article(topic):
    """Manage the article generation process and return the result."""
    st.write("🔍 Searching for relevant research papers...")

    references = search_research_papers(topic)

    if not references:
        st.error("❌ No relevant research papers found!")
        return
    
    num_references = st.slider(
        "📚 Select the number of references to use:",
        1,
        len(references),
        min(10, len(references)),  
    )

    selected_references = references[:num_references]

    prompt = generate_prompt(topic, selected_references)

    with st.spinner("✍️ Generating your article..."):
        response = model.invoke([{"role": "user", "content": prompt}])
   
    st.subheader("📑 Generated Technical Article")
    st.markdown(response.content)

if st.button("🚀 Generate Article") and topic:
    generate_article(topic)
