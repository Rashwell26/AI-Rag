from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


# 1. Connect to LLM
llm = OllamaLLM(model="tinyllama")

# 2. Load embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Load your document
loader = TextLoader("data.txt")
documents = loader.load()

# 4. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

# 5. Create vector database
db = Chroma.from_documents(docs, embeddings)

# 6. Ask question
query = input("Ask a question: ")

retrieved_docs = db.similarity_search(query)

context = "\n".join([doc.page_content for doc in retrieved_docs])

response = llm.invoke(
    f"Use this context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
)

print("\nAnswer:\n")
print(response)
