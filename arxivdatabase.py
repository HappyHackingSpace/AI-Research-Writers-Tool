from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import arxiv
import time
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma(
    collection_name="arxiv_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_arxiv_db",
)

categories = {
    "Artificial Intelligence": "cs.AI",
    "Computation and Language": "cs.CL",
    "Computational Complexity": "cs.CC",
    "Computational Engineering, Finance, and Science": "cs.CE",
    "Computational Geometry": "cs.CG",
    "Computer Science and Game Theory": "cs.GT",
    "Computer Vision and Pattern Recognition": "cs.CV",
    "Computers and Society": "cs.CY",
    "Cryptography and Security": "cs.CR",
    "Data Structures and Algorithms": "cs.DS",
    "Databases": "cs.DB",
    "Digital Libraries": "cs.DL",
    "Discrete Mathematics": "cs.DM",
    "Distributed, Parallel, and Cluster Computing": "cs.DC",
    "Emerging Technologies": "cs.ET",
    "Formal Languages and Automata Theory": "cs.FL",
    "General Literature": "cs.GL",
    "Graphics": "cs.GR",
    "Hardware Architecture": "cs.AR",
    "Human-Computer Interaction": "cs.HC",
    "Information Retrieval": "cs.IR",
    "Information Theory": "cs.IT",
    "Logic in Computer Science": "cs.LO",
    "Machine Learning": "cs.LG",
    "Mathematical Software": "cs.MS",
    "Multiagent Systems": "cs.MA",
    "Multimedia": "cs.MM",
    "Networking and Internet Architecture": "cs.NI",
    "Neural and Evolutionary Computing": "cs.NE",
    "Numerical Analysis": "cs.NA",
    "Operating Systems": "cs.OS",
    "Other Computer Science": "cs.OH",
    "Performance": "cs.PF",
    "Programming Languages": "cs.PL",
    "Robotics": "cs.RO",
    "Social and Information Networks": "cs.SI",
    "Software Engineering": "cs.SE",
    "Sound": "cs.SD",
    "Symbolic Computation": "cs.SC",
    "Systems and Control": "cs.SY",
}

num_papers = 5000
results_article = 200

client = arxiv.Client()
all_papers_dict = {}


def fetch_arxiv_papers(category_name, category_code, max_results):
    papers = []
    fetched = 0

    while fetched < max_results:
        search = arxiv.Search(
            query=f"cat:{category_code}",
            max_results=min(results_article, max_results - fetched),
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        results = client.results(search)

        for result in results:
            paper_id = result.entry_id

            if paper_id in all_papers_dict:
                if category_name not in all_papers_dict[paper_id]["categories"]:
                    all_papers_dict[paper_id]["categories"].append(category_name)
            else:
                all_papers_dict[paper_id] = {
                    "id": paper_id,
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "summary": result.summary,
                    "date": result.published.strftime("%Y-%m-%d"),
                    "link": result.entry_id,
                    "categories": [category_name],
                }

            fetched += 1

            if fetched >= max_results:
                break

        print(f"{category_name}: {fetched}/{max_results} makale çekildi.")
        time.sleep(5)


for category_name, category_code in categories.items():
    print(f"Fetching papers for category: {category_name} ({category_code})")
    fetch_arxiv_papers(category_name, category_code, num_papers)


all_papers = list(all_papers_dict.values())

with open("arxiv_papers.json", "w", encoding="utf-8") as f:
    json.dump(all_papers, f, indent=4, ensure_ascii=False)

print("Makale verisi 'arxiv_papers.json' dosyasına kaydedildi.")


docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

for paper in all_papers:
    text = f"{paper['title']}\n\n{paper['summary']}"
    chunks = text_splitter.split_text(text)

    for chunk in chunks:
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "title": paper["title"],
                    "authors": paper["authors"],  
                    "summary": paper["summary"],  
                    "categories": ", ".join(paper["categories"]),
                    "link": paper["link"],
                    "date": paper["date"],
                },
            )
        )


batch_size = 5000

for i in range(0, len(docs), batch_size):
    batch = docs[i : i + batch_size]
    vector_db.add_documents(batch)
    print(f"{i + len(batch)}/{len(docs)} doküman eklendi.")

num_docs = vector_db._collection.count()
print(f"Vektör veritabanındaki doküman sayısı: {num_docs}")
