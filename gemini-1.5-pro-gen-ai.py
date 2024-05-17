import google.generativeai as genai
import os
from dotenv import load_dotenv
import textwrap
import numpy as np
import pandas as pd
import google.ai.generativelanguage as glm

# Load environment variables from .env file
load_dotenv()

# Configure generative AI with the API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Generate content using the GenerativeModel
model = genai.GenerativeModel("gemini-1.0-pro-latest")
response = model.generate_content("The opposite of hot is")
print(response.text)

# Define title and sample text for embedding
title = "The next generation of AI for developers and Google Workspace"
sample_text = (
    "Title: The next generation of AI for developers and Google Workspace\n"
    "Full article:\n\n"
    "Gemini API & Google AI Studio: An approachable way to explore and prototype with generative AI applications"
)

# Define the model for embedding
embedding_model = "models/embedding-001"

# Generate embedding for the sample text
embedding = genai.embed_content(
    model=embedding_model,
    content=sample_text,
    task_type="retrieval_document",
    title=title,
)
# Uncomment to print the embedding
# print(embedding)

# Sample documents for embedding
DOCUMENT1 = {
    "title": "Operating the Climate Control System",
    "content": (
        "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. "
        "To operate the climate control system, use the buttons and knobs located on the center console. "
        "Temperature: The temperature knob controls the temperature inside the car. "
        "Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. "
        "Airflow: The airflow knob controls the amount of airflow inside the car. "
        "Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. "
        "Fan speed: The fan speed knob controls the speed of the fan. "
        "Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. "
        "Mode: The mode button allows you to select the desired mode. "
        "The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. "
        "Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. "
        "Defrost: The car will blow warm air onto the windshield to defrost it."
    ),
}
DOCUMENT2 = {
    "title": "Touchscreen",
    "content": (
        "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. "
        'To use the touchscreen display, simply touch the desired icon. For example, you can touch the "Navigation" icon to get directions to your destination '
        'or touch the "Music" icon to play your favorite songs.'
    ),
}
DOCUMENT3 = {
    "title": "Shifting Gears",
    "content": (
        "Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position. "
        "Park: This position is used when you are parked. The wheels are locked and the car cannot move. "
        "Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. "
        "The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. "
        "Low: This position is used for driving in snow or other slippery conditions."
    ),
}

# Create a DataFrame with the documents
documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
df = pd.DataFrame(documents)
df.columns = ["Title", "Text"]
print(df)


# Function to generate embeddings for a document
def embed_fn(title, text):
    return genai.embed_content(
        model=embedding_model, content=text, task_type="retrieval_document", title=title
    )["embedding"]


# Add embeddings to the DataFrame
df["Embeddings"] = df.apply(lambda row: embed_fn(row["Title"], row["Text"]), axis=1)
print(df)

# Define a query for document retrieval
query = "How do you shift gears in the Google car?"


# Function to find the best matching passage for the query
def find_best_passage(query, dataframe):
    """
    Compute the distances between the query and each document in the dataframe
    using the dot product.
    """
    query_embedding = genai.embed_content(
        model=embedding_model, content=query, task_type="retrieval_query"
    )
    dot_products = np.dot(
        np.stack(dataframe["Embeddings"]), query_embedding["embedding"]
    )
    idx = np.argmax(dot_products)
    return dataframe.iloc[idx]["Text"]  # Return text from index with max value


# Find the best passage matching the query
passage = find_best_passage(query, df)
print(passage)


# Function to create a prompt for the generative model
def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent(
        f"""
        You are a helpful and informative bot that answers questions using text from the reference passage included below. 
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
        strike a friendly and conversational tone. 
        If the passage is irrelevant to the answer, you may ignore it.
        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

        ANSWER:
        """
    )
    return prompt


# Create the prompt for the generative model
prompt = make_prompt(query, passage)
print(prompt)

# List available models and their supported generation methods
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)

# Generate the answer using the generative model
generative_model = genai.GenerativeModel("models/gemini-pro")
answer = generative_model.generate_content(prompt)
print(answer.text)
