import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

# Download necessary resources for NLP processing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Clean text data for processing
def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize words
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]

    # Rejoin words into a cleaned sentence
    return " ".join(cleaned_tokens)

# Apply cleaning function to the 'Review' column
def preprocess_data(df):
    df["Cleaned_Review"] = df["Review"].astype(str).apply(clean_text)
    return df

# Convert reviews into embeddings using a HuggingFace model
def create_embeddings(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = [embedding_model.embed_query(text) for text in documents]
    return embeddings

# Create and save a FAISS vector store
def create_faiss_index(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(documents, embedding_model)
    vector_store.save_local("faiss_reviews_index")
    return vector_store

# Set up the LLM (using OpenAI API, for example)
def create_llm_chain():
    llm = OpenAI(temperature=0)
    
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    llm_chain = LLMChain(llm=llm, prompt=PROMPT)
    
    return llm_chain

# Create RetrievalQA chain using FAISS and the LLM
def create_retrieval_qa_chain(vector_store, llm_chain):
    retriever = vector_store.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm_chain.llm, 
        chain_type="stuff", 
        retriever=retriever,
        chain_type_kwargs={"prompt": llm_chain.prompt}
    )
    return rag_chain

# Query the model
def query_model(rag_chain, query):
    response = rag_chain.run(query)
    return response

# Main function to execute the summarizer app
def main():
    # Load and preprocess the data
    df = load_data("/content/music_album_reviews.csv")
    df = preprocess_data(df)
    
    # Save cleaned data
    df.to_csv("cleaned_music_album_reviews.csv", index=False)
    print("Text cleaning complete! Saved as cleaned_music_album_reviews.csv")
    
    # Create FAISS vector store
    vector_store = create_faiss_index(df["Cleaned_Review"].tolist())

    # Create LLMChain and RetrievalQA chain
    llm_chain = create_llm_chain()
    rag_chain = create_retrieval_qa_chain(vector_store, llm_chain)
    
    # Example query
    query = "What do people think about OK Computer?"
    response = query_model(rag_chain, query)
    
    print("RAG Response:", response)

if __name__ == "__main__":
    main()
