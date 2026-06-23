from pathlib import Path
import pandas as pd
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

env_path = Path(__file__).parent.parent / "Resource" / ".env"
load_dotenv(env_path )

faqs_path = Path(__file__).parent.parent/"Resource"/"faq_data.csv"
chroma_client = chromadb.Client()
collection_name_faq = 'faqs'
groq_client = Groq()

def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print("Ingesting faq data in Chromadb...")
        collection = chroma_client.get_or_create_collection(
            collection_name_faq
        )
        df=pd.read_csv(path)
        docs=df['question'].to_list()
        matadata= [{'answer':ans} for ans in df['answer'].to_list()]
        ids= [f"id_{i}" for i in range (len(docs))]

        collection.add(
           documents=docs,
           metadatas=matadata,
           ids=ids
               )
        print(f"FAQ Data succcessfully ingested in chroma collection {collection_name_faq}")
    else:
       print(f"Collection {collection_name_faq} already exists")

def get_relevant_qa(query):
    collection = chroma_client.get_collection(collection_name_faq)
    result= collection.query(
        query_texts=[query],
        n_results=2
    )
    return result

def faq_chain(query):
    result = get_relevant_qa(query)
    context = "\n".join(
        [r["answer"] for r in result["metadatas"][0]]
    )

    answer = generate_answers(query,context)
    return answer

def generate_answers(query, context):
    prompt = f""" Given the question and context below. generate the nswer based on the context only.
    if you don't find the answer inside the context then ssy "I don't know".
    Do not make things up.
    
    
    QUESTION : {query}
    
    CONTEXT : {context}
    """

 

    print("\nContext:")
    print(context)

    #call llm
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ],
        model= os.environ['GROQ_MODEL'],
        temperature=0
    )

    return chat_completion.choices[0].message.content

if __name__ == "__main__":
   ingest_faq_data(faqs_path)
   query = "what's your policy on defective products"
   result = get_relevant_qa(query)
   answer = faq_chain(query)
   print(result["documents"])
   print(result["metadatas"])
   print(answer)
