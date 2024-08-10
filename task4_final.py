import os
import streamlit as st 
import boto3
import sqlite3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Attr
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, load_tools
from pyowm import OWM
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain_community.vectorstores import FAISS
from PIL import Image

'''Task 4: Create an end-to-end workflow program'''

st.title("Hello, I am HODAML Team 1 Chatbot!")

PDF_FOLDER = 'docs/'  # Update with the actual folder path 
DOCUMENTS_TITLES = ['TRexSafeTemp.pdf', 'VelociraptorsSafeTemp.pdf']

os.environ["OPENAI_API_KEY"] = 'example_key'
OPENWEATHERMAP_API_KEY = "example_key"
AWS_ACCESS_KEY_ID = 'example_key' # Update with your Access Key
AWS_SECRET_ACCESS_KEY = 'example_key' #Update with your secret access key
REGION_NAME = 'us-east-1'

# Initialize session state variables in Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

if "generate_email" not in st.session_state:
    st.session_state["generate_email"] = False

def get_city_and_dinoId(date):
    """Retrieve the city and dinoId from the table given the date"""
    dynamodb = boto3.resource('dynamodb',
                            aws_access_key_id=AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                            region_name=REGION_NAME)

    table = dynamodb.Table("DynoTransport")  # Update with your table name in DynamoDB

    try:
        response = table.scan(
            TableName="DynoTransport",
            FilterExpression=Attr('Date').eq(date)
        )

    except ClientError as e:
        raise Exception
    else:
        return response.get('Items', [])

def get_name_by_dinoid(id):
    """Get the name of the dinosaur given the DinoID"""

    # Connect to the SQLite database
    conn = sqlite3.connect("database/dino.db")
    cursor = conn.cursor()

    try:
        # Execute the query to retrieve the name based on the provided ID
        cursor.execute("SELECT name FROM DinoMap WHERE id = ?", (id,))
        # Fetch the result
        result = cursor.fetchone()

        if result:
            # If a name is found, return it
            return result[0]
        else:
            # If no matching ID is found, return None
            return None

    finally:
        # Close the database connection
        conn.close()

def get_current_city_temperature_forecast(city):
    """Get the current temperature of a city"""
    owm = OWM(OPENWEATHERMAP_API_KEY)
    mgr = owm.weather_manager()
    # Search for current weather in a specific city and get details
    observation = mgr.weather_at_place(city)
    w = observation.weather
    return w.temperature('fahrenheit')["temp"]

def send_text_message(phone_number, message):
    """Send text message to a given a phone number"""
    sns = boto3.client('sns', region_name=REGION_NAME)

    # Replace with your AWS account ID
    aws_access_key_id = AWS_ACCESS_KEY_ID

    # Create the SNS topic ARN (Amazon Resource Name)
    topic_arn = f'arn:aws:sns:us-east-1:637423238992:dso599_project'

    # Format the phone number with the international prefix
    formatted_phone = f'+1{phone_number[-10:]}'

    # Publish the message to the SNS topic
    sns.publish(
        PhoneNumber=formatted_phone,
        Message=message,
        MessageAttributes={
            'AWS.SNS.SMS.SMSType': {
                'DataType': 'String',
                'StringValue': 'Transactional'
            }
        }
    )
    return f'Text message about the dinosaur status sent to {formatted_phone}'

def create_retrieval_agent_executor():
    # Prepare documents to load dinasours' information
    loader = PyPDFDirectoryLoader(PDF_FOLDER, recursive=True)
    documents = loader.load()
    filtered_documents = [doc for doc in documents if any(title in doc.metadata['source'] for title in DOCUMENTS_TITLES)]

    ### Chunking 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Use a sentence transformer model to generate embeddings for the chunks
    embeddings = OpenAIEmbeddings() 
    db = FAISS.from_documents(texts, embeddings)

    ### Convert the vector store into a retriever that can fetch relevant document snippets based on queries
    retriever = db.as_retriever()

    ### Create a tool for our retriever
    tool = create_retriever_tool(
        retriever,
        "search_information_from_vector_store",
        "Searches and returns excerpts from the TRexSafeTemp.pdf and VelociraptorsSafeTemp.pdf.",
    )
    tools = [tool]

    # Handle the question-answering format
    prompt = hub.pull("hwchase17/openai-tools-agent") 

    ### Initialize the model

    llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo") 
    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools)

def display_previous_messages():
    if len(st.session_state.messages) == 0:
        return
    
    with st.chat_message("assistant"):
        for message in st.session_state.messages:
            st.markdown(message["content"])

def generate_email_button():
    st.session_state["generate_email"] = True

def chat_input_submission():
    st.session_state.messages = []
    st.session_state["generate_email"] = False

def display_and_save_assistant_output(output):
    st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})

def main_chat():
    # Accept user input
    agent_executor = create_retrieval_agent_executor()
    dino_name = ""

    if date := st.chat_input("Please enter the date to lookup", on_submit = chat_input_submission):
        with st.chat_message("assistant"):
            # Get city and dinoId by date
            response = get_city_and_dinoId(date)

            if response == []:
                st.markdown(f"Cannot find any dinosaur transported on {date}. Please type in another date to search for!")
                return
            
            dino_id = response[0]["DinoID_Transported"]
            dino_name = get_name_by_dinoid(dino_id)
            city = response[0]["City"]

            display_and_save_assistant_output(f"The dinosaur transported on {date} is a {dino_name} with DinoID {dino_id} being transported from {city}")
            
            # Get temperature based on city
            temperature = get_current_city_temperature_forecast(city)
            display_and_save_assistant_output(f"The current temperature in {city} is {temperature}°F")

            # Check if temperature is safe
            is_temperature_safe = agent_executor.invoke({"input": f"is {temperature}F a safe temperature for the {dino_name}?. Return only the bool output"})
            if is_temperature_safe == "true":
                display_and_save_assistant_output(f"The current temperature of {temperature}°F in {city} is safe for the {dino_name}")
                st.session_state["generate_email"] = True
            else:
                display_and_save_assistant_output(f"Please pay attention! The current temperature of {temperature}°F in {city} is not safe for the {dino_name}")

                plan = agent_executor.invoke({"input": f"Here are some actions needed to keep the {dino_name} safe when the temperature is outside the range:"})
                display_and_save_assistant_output(plan["output"])

                # Create button to generate email in Streamlit
                st.button("Generate Status Email to Management Team", on_click=generate_email_button)

    if st.session_state["generate_email"]:
        with st.chat_message("assistant"):
            email_status = agent_executor.invoke({
                "input": f"Craft an email to the management giving them a status report of {dino_name} on your situation. I want to include the current status, challenges, and next steps in my email."
            })
            st.markdown(email_status["output"])

        # Send text message to a phone number
        text_message = 'Hello, this is an update about the dinosaur from Team 1 HODAML AWS SNS service'       
        message_sent = send_text_message('4133133089',text_message)
        st.markdown(message_sent)

        # Include a picture of a text message sent
        left_co, cent_co,last_co = st.columns(3)
        image = Image.open('Text_Screenshot.jpg')
        image = image.resize((400, 800))
        with cent_co:
            st.image(image, caption='Text message sent')

if __name__ == '__main__':
    display_previous_messages()
    main_chat()