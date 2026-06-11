import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Loading PDF document...")
loader = PyPDFLoader("TechCorp_Official_Employee_Handbook.pdf")
document = loader.load()


print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(document)


print("Creating vector database...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)
"""
user_question = "What days can I work from home?"
print(f"\nQuestion: {user_question}")

response = rag_chain.invoke(user_question)
# print(f"Answer: {response.content}")

# Clean up the output if Gemini returns a list of content blocks
if isinstance(response.content, list):
    clean_answer = response.content[0]['text']
else:
    clean_answer = response.content

print(f"Answer: {clean_answer}")
"""


print("\n--- PDF Chatbot Initialized ---")
print("Type 'exit' or 'quit' to stop.")

def chat_with_pdf(message, history):

    response = rag_chain.invoke(message)

    if isinstance(response.content, list):
        clean_answer = response.content[0]['text']
    else:
        clean_answer = response.content

    return clean_answer

import gradio as gr

demo = gr.ChatInterface(
    fn=chat_with_pdf,
    title="PDF RAG Chatbot",
    description="Ask questions about your employee handbook."
)

demo.launch(share = True)
