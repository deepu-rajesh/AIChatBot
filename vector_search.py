import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import openai
from openai import Completion

# Initialize Pinecone
pinecone_api_key = ''
pinecone_environment = 'us-east-1'

pc = pinecone.Pinecone(api_key=pinecone_api_key)
index_name = "chatbot-index"

# Check if the index exists and create if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches the dimension of your embeddings
        metric='cosine',  # You can choose 'cosine', 'euclidean', etc. based on your use case
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region=pinecone_environment
        )
    )
index = pc.Index(index_name)

def add_chunks_to_index(text):
    """Add text chunks to the Pinecone index."""
    # Ensure text is a single string
    if isinstance(text, list):
        text = ' '.join(text)  # Join list elements into a single string
    
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)  # Smaller chunk size with overlap
    text_chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    chunk_embeddings = embeddings.embed_documents(text_chunks)
    
    for text_chunk, embedding in zip(text_chunks, chunk_embeddings):
        text_id = str(hash(text_chunk))
        index.upsert(
            vectors=[{
                'id': text_id,
                'values': embedding,
                'metadata': {'text': text_chunk}
            }]
        )

# def vector_search(query):
#     # Generate the vector for the query
#     query_vector = OpenAIEmbeddings(openai_api_key=openai.api_key).embed_documents([query])[0]
    
#     # Query the Pinecone index with the generated vector
#     response = index.query(
#         vector=query_vector,
#         top_k=5,  # Retrieve the top 5 matches
#         include_values=True,
#         include_metadata=True,
#     )
    
#     # Handle case where no matches are found
#     if 'matches' not in response or len(response['matches']) == 0:
#         return ["No relevant results found."]
    
#     # Extract and return relevant information based on a similarity threshold
#     similarity_threshold = 0.8  # Adjust the threshold as needed
#     relevant_texts = [match['metadata']['text'] for match in response['matches'] if match['score'] > similarity_threshold]
    
#     if not relevant_texts:
#         return ["No relevant results found."]

#     return relevant_texts

def vector_search(query):
    # Generate the vector for the query
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    query_vector = embeddings.embed_documents([query])[0]
    
    # Query the Pinecone index with the generated vector
    response = index.query(
        vector=query_vector,
        top_k=5,  # Retrieve the top 5 matches
        include_values=True,
        include_metadata=True,
    )
    
    # Handle case where no matches are found
    if 'matches' not in response or len(response['matches']) == 0:
        return ["No relevant results found."]
    
    # Extract relevant information based on a similarity threshold
    similarity_threshold = 0.8  # Adjust the threshold as needed
    relevant_texts = [match['metadata']['text'] for match in response['matches'] if match['score'] > similarity_threshold]
    
    if not relevant_texts:
        return ["No relevant results found."]
    
    # Combine the relevant texts into a single context
    context = " ".join(relevant_texts)
    
    # Generate a response using the chat model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the appropriate model name
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    
    # Extract and return the generated answer
    answer = response['choices'][0]['message']['content'].strip()
    
    return [answer]