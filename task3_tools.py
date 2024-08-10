import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
import pandas as pd
import streamlit as st
import os
import openai
import requests 
import sqlite3


def create_dynamodb_table():
    # Initialize a boto3 client for DynamoDB
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

    # Create the DynamoDB table
    table = dynamodb.create_table(
        TableName='DinoTransport',
        KeySchema=[
            {'AttributeName': 'Date', 'KeyType': 'HASH'},  # Partition key
            {'AttributeName': 'Route_Number', 'KeyType': 'RANGE'}  # Sort key
        ],
        AttributeDefinitions=[
            {'AttributeName': 'Date', 'AttributeType': 'S'},
            {'AttributeName': 'Route_Number', 'AttributeType': 'N'}
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 1,
            'WriteCapacityUnits': 1
        }
    )

    # Wait for the table to be created
    table.wait_until_exists()
    return "Table created successfully."

# Function to read CSV and upload data to DynamoDB
def upload_data_from_csv(csv_file_path):
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('DinoTransport')
    data = pd.read_csv(csv_file_path)

    # Uploading data to the table
    with table.batch_writer() as batch:
        for index, row in data.iterrows():
            batch.put_item(
                Item={
                    'Route_Number': int(row['Route_Number']),
                    'Date': row['Date'],
                    'City': row['City'],
                    'DinoID_Transported': row['DinoID_Transported']
                }
            )
    return "Data uploaded successfully."

# Path to CSV file
CSV_FILE_PATH = '/Users/faeez.aroos/Downloads/data.csv'

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('DinoTransport')

DATABASE_PATH = '/Users/faeez.aroos/Downloads/dino.db'

def create_sqlite_table():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS DinoMap (
            DinoID TEXT PRIMARY KEY,
            DinoName TEXT NOT NULL
        )
    ''')
    dino_data = [
        ('T88', 'T-Rex'),
        ('V66', 'Velociraptor')
    ]
    cursor.executemany('INSERT OR IGNORE INTO DinoMap (DinoID, DinoName) VALUES (?, ?)', dino_data)
    conn.commit()
    conn.close()

create_sqlite_table()

def retrieve_dino_data(date):
    response = table.query(
        KeyConditionExpression=Key('Date').eq(date)
    )
    return [{'City': item['City'], 'DinoID_Transported': item['DinoID_Transported']} for item in response['Items']]

def get_dino_name(dino_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT DinoName FROM DinoMap WHERE DinoID = ?', (dino_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Dino ID not found"

# Set up your Streamlit interface
st.title('Dinosaur Transport Details')

# Date input
date_input = st.text_input('Enter the date (MM/DD/YYYY):')

# Button to fetch data
if st.button('Retrieve Data'):
    if date_input:
        # Fetch data from DynamoDB
        results = retrieve_dino_data(date_input)
        if results:
            for result in results:
                dino_name= get_dino_name(result['DinoID_Transported'])
                st.write(f"City: {result['City']}, DinoID: {result['DinoID_Transported']}, Dino Name: {dino_name}")
        else:
            st.write("No data found for this date.")
    else:
        st.write("Please enter a valid date.")

openai_api_key = st.secrets["openai"]["api_key"]

def generate_description(date):
    openai.api_key = st.secrets["openai"]["api_key"]  # Set the API key here or ensure it's set globally
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Updated to use the latest model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Generate a detailed description for dinosaur transport data on {date}"}
            ]
        )
        return response['choices'][0]['message']['content']  # Adjusted to the new response format
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_current_temperature(city):
    api_key = "example_key"  # Replace with your actual API key
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=imperial"
    
    response = requests.get(url)
    if response.status_code == 200:  # API request is successful
        data = response.json()
        temperature = data['main']['temp']
        return f"Current temperature in {city} is {temperature}°F"
    else:
        return "Failed to retrieve temperature"

get_current_temperature("London")

st.title('Weather Information')

city = st.text_input('Enter the city to get the current temperature:')

if st.button('Get Temperature'):
    if city:
        temperature_info = get_current_temperature(city)
        st.write(temperature_info)
    else:
        st.write("Please enter a valid city name.")
