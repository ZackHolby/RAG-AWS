from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


DATA_PATH="./data/GOT.md"
CHROMA_PATH="chroma"

def load_docs():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    #print some relevant info
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    random_doc = chunks[2]
    print("RANDOM DOC INFO")
    print(random_doc.page_content)
    print(random_doc.metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def load_db():
    docs = load_docs()
    chunks = split_docs(docs)
    save_to_chroma(chunks)

##run all code and load docs into Chroma DB
load_db()


