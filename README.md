Interactive AI Chatbot Development
---
**Project Overview**
- This project demonstrates the development of an AI-driven chatbot and workflow automation system using a combination of advanced technologies and frameworks. The system leverages OpenAI's language models for natural language processing, Streamlit for the user interface, and LangChain for creating AI agents and tools. It integrates with AWS services like DynamoDB and SNS, and external APIs like OpenWeatherMap.

**Project Goals and Key Components**

1. **Interactive Chatbot Creation**:
   - Utilize Streamlit and OpenAI's language models to create an interactive chatbot interface for user interaction.

2. **Vector-Based Document Retrieval System**:
   - Stored and retrieved information from documents using a ReAct agent. This task involved answering specific questions and crafting emails based on the retrieved data.

3. **Tool and Function Development**:
   - Develop tools and functions to interact with various databases (DynamoDB, SQLite) and external APIs (OpenWeatherMap).

4. **End-to-End Workflow Construction**:
   - **Information Retrieval**: Access information about dinosaur transportation based on a given date.
   - **Temperature Check**: Verify the current temperature at the transportation location.
   - **Safety Determination**: Assess if the temperature is safe for the specific dinosaur.
   - **Safety Recommendations**: Provide safety recommendations if needed.
   - **Status Updates**: Generate and send status update emails to management.
   - **Text Messaging**: Send text messages with relevant information.

5. **AI Agents for Orchestration**:
   - Use AI agents (via the LangChain framework) to coordinate various tools and make decisions based on the retrieved information.

6. **Practical Application of LLMs**:
   - Showcase the practical use of Large Language Models (LLMs) in a real-world scenario that combines data retrieval, decision making, and automated communication.
