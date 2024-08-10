import os
import streamlit as st 

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

### Set the title of the Streamlit web interface
st.title("Hi, I am _HODAML_, ask me something about the pdf") 

### Load the files in the folder
pdf_folder = '/Users/graceli/Desktop/599 AWS/project2_task2/'  # Update with the actual folder path 
loader = PyPDFDirectoryLoader(pdf_folder, recursive=True)
documents = loader.load()

### Retrieve the specific documents
document_titles = ['TRexSafeTemp.pdf', 'VelociraptorsSafeTemp.pdf']
filtered_documents = [doc for doc in documents if any(title in doc.metadata['source'] for title in document_titles)]

### Chunking 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
# Use a sentence transformer model to generate embeddings for the chunks

embeddings = OpenAIEmbeddings(openai_api_key="example_key")
### Create a FAISS vector store from the document embeddings
db = FAISS.from_documents(texts, embeddings)

### Convert the vector store into a retriever that can fetch relevant document snippets based on queries
retriever = db.as_retriever()

### create a tool for our retriever
tool = create_retriever_tool(
    retriever,
    "search_information_from_vector_store",
    "Searches and returns excerpts from the TRexSafeTemp.pdf and VelociraptorsSafeTemp.pdf.",
)
tools = [tool]

# handle the question-answering format
prompt = hub.pull("hwchase17/openai-tools-agent") 

### Initialize the model

llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo", openai_api_key="example_key")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Query text
user_question = st.text_input('How can I help you today?', "")

# Process the input using agent and display the response
if user_question:
        response = agent_executor.invoke({"input": user_question})
        with st.chat_message("assistant"):
              st.write(response['output'])

# Display the answer source
if st.checkbox("Show Document Sources"):
    st.write("Document Sections:")
    st.write([doc.metadata['source'] for doc in filtered_documents])
