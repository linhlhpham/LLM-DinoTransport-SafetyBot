import boto3 # type: ignore
import csv
import os
from botocore.exceptions import ClientError # type: ignore
from boto3.dynamodb.conditions import Key # type: ignore
from boto3.dynamodb.conditions import Attr # type: ignore
from langchain_openai import ChatOpenAI # type: ignore
from langchain import hub # type: ignore
from langchain.agents import create_tool_calling_agent # type: ignore
from langchain.tools import BaseTool, StructuredTool, tool # type: ignore
from langchain.agents import AgentExecutor # type: ignore
import sqlite3
from langchain_community.utilities import SQLDatabase # type: ignore
from langchain_community.agent_toolkits import create_sql_agent # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory

AWS_ACCESS_KEY_ID = 'example_key' #Your Access Key
AWS_SECRET_ACCESS_KEY = 'example_key+3SLtgkTnpQlJqB923073o' #Your secret access key
REGION_NAME = 'us-east-1'
os.environ["OPENAI_API_KEY"] = 'example_key'
os.environ["OPENWEATHERMAP_API_KEY"] = "example_key"

def create_dynamodb_table_and_upload_data(csv_file, table_name):
    # Create a DynamoDB client
    dynamodb = boto3.client('dynamodb',
                            aws_access_key_id = AWS_ACCESS_KEY_ID,
                            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
                            region_name = REGION_NAME)

    # Create a DynamoDB table
    dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {'AttributeName': 'Route_Number', 'KeyType': 'HASH'},  # Partition key
            {'AttributeName': 'Date', 'KeyType': 'RANGE'}          # Sort key
        ],
        AttributeDefinitions=[
            {'AttributeName': 'Route_Number', 'AttributeType': 'N'},
            {'AttributeName': 'Date', 'AttributeType': 'S'}
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )

    # Wait until the table exists
    dynamodb.get_waiter('table_exists').wait(TableName=table_name)

    # Upload data from the CSV file to DynamoDB
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dynamodb.put_item(
                TableName=table_name,
                Item={
                    'Route_Number': {'N': row['Route_Number']},
                    'Date': {'S': row['Date']},
                    'City': {'S': row['City']},
                    'DinoID_Transported': {'S': row['DinoID_Transported']}
                }
            )
    print(f"Data uploaded successfully to DynamoDB table: {table_name}")


@tool
def get_city_and_dinoId(date):
    """retrieve the city and dinoId from the table given the date"""
    dynamodb = boto3.resource('dynamodb',
                            aws_access_key_id=AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                            region_name=REGION_NAME)

    table = dynamodb.Table("DynoTransport")

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
    """retrieve the name given the dinoid"""
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


def llm_invoke_get_city_and_dinoId(date):
    tools = [get_city_and_dinoId]
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": f"What is the DynoID and City for {date}?"})

    return response["output"]

def llm_invoke_get_name_by_dinoid():
    db = SQLDatabase.from_uri("sqlite:///database/dino.db")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    temp = llm_invoke_get_city_and_dinoId("3/19/2024")
    agent_executor.invoke(
        f"Give me the Dino name where Dino ID is equal to the Id here: {temp}"
    )

def llm_invoke_get_current_temp(city):
    memory = ConversationBufferWindowMemory(memory_key="chat_history",k=5)

    # Initialize base LLM
    llm = ChatOpenAI(temperature=0)

    # Load tools
    tools = load_tools(["openweathermap-api"], llm=llm)

    # Initialize the agent with multiple tools
    agent_chain = initialize_agent(tools,
                                llm,
                                verbose=True,
                                memory=memory,
                                handle_parsing_errors=True)

    # Example invocation
    response = agent_chain.invoke(input=f"What is the current temperature in {city} now?")
    print("Weather response:", response["output"])

def send_text_message(phone_number, message):
    # Create an SNS client
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
    print(f'Text message sent to {formatted_phone}')

if __name__ == '__main__':
    llm_invoke_get_city_and_dinoId('3/19/2024')
    llm_invoke_get_name_by_dinoid()
    llm_invoke_get_current_temp('New York')
    send_text_message('2048005775','Hello, this is a test message from Team 1 HODAML AWS SNS service')