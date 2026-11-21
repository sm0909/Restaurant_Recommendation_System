"""
Structure Text and Multimodal Data with LLMs

Standalone Python version of the original analysis notebook.

Dependencies:
    pip install ibm-watsonx-ai==1.4.7 matplotlib==3.10.7 numpy==2.3.4

Configuration (environment variables, no values hardcoded):
    CALIFORNIA_CULINARY_MAP_URL  - source URL for California-Culinary-Map.txt
    LLM_MODEL_ID  - model identifier for your LLM provider
    LLM_PROJECT_ID  - project / workspace id for your LLM provider
    DATASET_URL  - set to your service location
"""

import os
import shutil
import tarfile
import urllib.request
import zipfile

# ---- Configuration -------------------------------------------------
# Values are read from the environment so nothing is hardcoded.
CALIFORNIA_CULINARY_MAP_URL = os.environ.get("CALIFORNIA_CULINARY_MAP_URL", "")  # source URL for California-Culinary-Map.txt
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "")  # model identifier for your LLM provider
LLM_PROJECT_ID = os.environ.get("LLM_PROJECT_ID", "")  # project / workspace id for your LLM provider
DATASET_URL = os.environ.get("DATASET_URL", "")  # set to your service location


# ---- Data helpers --------------------------------------------------
def download_file(url, dest):
    """Download `url` to `dest` (skips if the file already exists)."""
    if not url:
        raise RuntimeError("No source URL configured - set the matching environment variable.")
    if os.path.exists(dest):
        print("Already present:", dest)
        return dest
    print("Downloading", dest, "...")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as handle:
        shutil.copyfileobj(response, handle)
    return dest


def extract_archive(archive_path, dest_dir="."):
    """Extract a .tar/.tar.gz/.zip archive into `dest_dir`."""
    os.makedirs(dest_dir, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(dest_dir)
    else:
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(path=dest_dir)
    print("Extracted", archive_path, "->", dest_dir)
    return dest_dir


def download_and_extract(url, dest_dir="."):
    """Download an archive and extract it into `dest_dir`."""
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, os.path.basename(url.split("?")[0]) or "dataset.tar")
    download_file(url, archive_path)
    return extract_archive(archive_path, dest_dir)

# ---- Lab: Structure Unstructured Restaurant Data with an LLM -----------

# ---- Scenario ----------------------------------------------------------

# ---- Background --------------------------------------------------------
# You are a Data Engineer at a leading AI startup building a next-generation Connoisseur Companion. Unlike traditional recommendation systems that rely on coarse 1–5 star ratings, the organization aims to understand the why behind user preferences, capturing the vibe of a restaurant, dietary considerations, and standout “hero dishes” hidden inside unstructured, multimodal data.
# ---- The challenge -----------------------------------------------------
# The organization has acquired a large, messy dataset spanning multiple modalities:
# Restaurant descriptions and user reviews (raw TXT)
# Food recipes with structured metadata and images (JSON + JPEG)
# User restaurant visit histories (JSON with URLs)
# However, this data is not immediately usable:
# 1. Unstructured text lacks a consistent schema, making search and retrieval inefficient.
# 2. Images and URLs are opaque to traditional data pipelines and must be transformed into searchable, quantitative representations.
# 3. The data spans multiple modalities with no unified representation.
# To build a high-quality, explainable recommendation engine, your first task is to transform this heterogeneous, multimodal data into a structured knowledge base. This includes extracting semantic signals from text, generating representations for images and URLs, and organizing everything into a coherent, query-friendly JSON format.

# ---- Objectives --------------------------------------------------------

# In this lab, you will write a Python program that will:
# Load raw restaurant descriptions and review data from unstructured TXT files
# Use a multimodal/LLM-based pipeline to extract structured attributes (e.g., cuisine type, ambiance, dietary options, and signature dishes)
# Convert the extracted information into a well-defined JSON schema suitable for indexing and search

# ---- Important: About the lab environment ------------------------------

# Please be aware that sessions for this lab environment are not persisted. Every time you connect to this lab, a new environment is created for you. Any data you may have saved in the earlier session would get lost. Plan to complete these labs in a single session, to avoid losing your data.

# ---- Screenshot requirement for this lab -------------------------------

# You will be prompted to take a screenshot and save it on your own device. You will need this screenshot either to answer graded quiz questions or to upload as your submission for the Final Project at the end of this course. You can use various free screen-grabbing tools or your operating system's shortcut keys to do this (for example, Alt+PrintScreen on Windows and Command+shift+4 on Mac).
# Note: The screenshot can be saved with either the .jpg or .png extension.

# ----

# ---- Set up the lab environment ----------------------------------------

# For this lab, you will be using the following libraries:
# numpy for numerical operations and handling array-based data during preprocessing and analysis
# matplotlib for basic data visualization and plotting, useful for inspecting distributions or intermediate results
# json for parsing, constructing, and serializing structured JSON representations extracted from unstructured text
# Credentials to securely authenticate with the watsonx.ai service
# ModelInference to invoke foundation models for text understanding and information extraction
# GenTextParamsMetaNames to configure text generation and extraction parameters
# ModelTypes and DecodingMethods to select the appropriate model and control inference behavior
# These libraries together enable you to transform raw restaurant description text into structured, machine-readable knowledge using GenAI-powered workflows.

# ---- Install required libraries ----------------------------------------
# Run the following code block to install all required libraries:

# ---- Import required libraries -----------------------------------------
# It is recommended to import all required libraries in one place (here):

import numpy as np
import matplotlib.pyplot as plt
import json
import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import (
    ModelTypes,
    DecodingMethods,
)

# Libraries and codes to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
print(warnings.filterwarnings('ignore'))

# ---- Fetch the data file -----------------------------------------------

# Run the following code to fetch the restaurant description text file:

print(download_file(CALIFORNIA_CULINARY_MAP_URL, "California-Culinary-Map.txt"))

# ----

# ---- Exercise 1: Load the data and display the texts -------------------

# Great! You now have the California-Culinary-Map.txt file in your folder. Before diving into the data, it is important to load and explore it first to gain a high-level understanding of its structure and content. This exercise gives you the opportunity to inspect the raw data and prepare it for transformation into a more structured, machine-accessible format.

# ---- Step 1: Load the data ---------------------------------------------
# In Step 1, you will load the text data file and explore the restaurant contents. Make sure you use the correct file path that links to the file California-Culinary-Map.txt.

### Your Code Here:
### 1.1: Define the file_path to the text file
# TODO (exercise): file_path = ...

### 1.2: Open the text file


### 1.3: Print the first 100 characters of the restaurant data

# ---- Step 2: Split the restaurant paragraphs into a Python list --------
# In the previous step, you may have noticed that the text file consists of multiple paragraphs, each describing a single restaurant. In Step 2, you will split these paragraphs into a list, allowing you to work with and manage each restaurant’s data more effectively.

### Your Code Here:
### 2.1: Split the restaurant paragraphs into list (hint: use .split)
# TODO (exercise): restaurant_list = ...

### 2.2: Since the first item is the dataset name, we remove it
restaurant_list = restaurant_list[1:]

### 2.3: Print out the number of restaurants we have


### 2.4: Print out the first item to have a closer look at the content

# ----

# ---- Exercise 2: Define the LLM ----------------------------------------

# You may have observed that each restaurant description includes the following key attributes:
# 1. Restaurant name
# 2. Location
# 3. Restaurant type
# 4. Food style
# 5. Rating
# 6. Price range
# 7. Signature dishes
# 8. Specialties
# 9. Shortcomings
# This information is essential for uniquely identifying and describing a restaurant.
# In this lab, you will use a large language model (LLM) to organize these attributes into a structured JSON format. As a first step, you will define the base LLM that will extract and structure the information from the unstructured text.

# ---- Step 1: Implement the LLM function --------------------------------

def llm_model(system_msg, prompt_txt):
    #system_msg: the system message given to the LLM
    #prompt_txt: the user prompt
    
    model_id = LLM_MODEL_ID

    project_id = LLM_PROJECT_ID

    credentials = Credentials(
                    url = DATASET_URL
                    )

    ### 1.1: Define the model by ModelInference

    ### 1.2: Define the messages

    ### 1.3: Get the final response output and return it

    pass

# ---- Step 2: Test your llm_model() -------------------------------------

# Test your LLM function with the following inputs:

import time

def safe_llm_call(system_msg, prompt_txt, retries=3):
    for i in range(retries):
        try:
            return llm_model(system_msg, prompt_txt)
        except Exception:
            time.sleep(2)
    return "Failed after retries"

system_msg = "You are a helpful assistant."
prompt_txt = "Which place is warmer in winter? Hawaii or Greenland?"
print(safe_llm_call(system_msg, prompt_txt))

# ----

# ---- Exercise 3: Prompt Engineering ------------------------------------

# To make an LLM behave reliably and produce outputs that meet your requirements, careful prompt design is essential. Here is a recap of key prompt engineering techniques:
# Clear and specific instructions: Explicitly state the task, expected output format, and constraints to reduce ambiguity.
# Structured output guidance: Provide schemas, templates, or examples (e.g., JSON formats) to encourage consistent, machine-readable responses.
# Few-shot prompting: Include representative input–output examples to guide the model toward the desired behavior.
# Role prompting: Assign the model a specific role (e.g., “You are a data extraction assistant”) to shape its reasoning and tone.
# Constraint-based prompting: Define what the model should and should not do to improve precision and reliability.
# Iterative refinement: Evaluate outputs and progressively refine prompts to improve performance over time.
# This exercise focuses on one-shot prompting, a special case of few-shot prompting. You will define a function that generates the prompt templates.

# ---- Step 1: Define the template ---------------------------------------

# In Step 1, you will implement a function that generates prompts for the LLM, given an input restaurant description paragraph.
# You will apply the one-shot prompting technique, which includes a single example of the desired output within the prompt to guide the model’s response.
# An example output is provided, corresponding to the second restaurant paragraph in your restaurant_list variable. Using this example, you will design a prompt template that enables the model to consistently transform any input restaurant description into the required structured JSON format.
# Note: For the price range field, instruct the LLM to convert dollar signs (e.g., \\\\\\\\\\\\\\\\\\\\\\$, \\$\\$, $$$) into an integer representing the number of dollar symbols.

EXAMPLE_RESTAURANT_PARAGRAPH = restaurant_list[1] #use the second restaurant paragraph as the example
EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier."
    "shortcomings": []
    }}
"""

### Design your prompt here
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    [YOUR PROMPT HERE]
    """
    
    base_user_prompt = f"""
    Task:
    [YOUR PROMPT HERE]

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    
    """
    return base_system_msg, base_user_prompt

# ---- Step 2: Test your prompts -----------------------------------------

# Run the following code to verify that your prompt produces the expected output:

# Unit test:
restaurant_paragraph = restaurant_list[0]
base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph=restaurant_paragraph)

test_response = llm_model(system_msg=base_system_msg, prompt_txt=base_user_prompt)
print(test_response)

# ---- Step 3: Validate the LLM outputs ----------------------------------

# LLMs are not always perfectly reliable and may produce errors, even when strict rules are specified in the prompt. To ensure that the generated outputs strictly conform to the required JSON format, you will learn how to define a function that formally validates the LLM output.
# Here, you are provided with the function and the test. You don't have to implement anything. Run the following code to validate the test response you obtained in Step 2.

# Validation
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

### 3.1. Define the schema
class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


### 3.2. Use the validation method to validate the test_response from the unit test
try:
    restaurant_data = Restaurant.model_validate_json(test_response)
    print(f"Success! Validated: {restaurant_data.name}")
except ValidationError as e:
    print(f"Validation failed: {e.json()}")

# ----

# ---- Exercise 4: Structure all the restaurant data ---------------------

# ---- Step 1: Define prompts to have an LLM that auto repairs outputs to JSON format ----

# As mentioned earlier, LLMs are not perfect. While you previously defined a schema to validate the output, validation alone does not fix incorrect results. To address this, introduce an additional LLM “expert” whose sole responsibility is to automatically repair and correct outputs that do not conform to the required JSON format.
# In this step, you will define the prompt template function for this LLM. This function takes:
# - candidate_json_output: The candidate json output
# - error_message: The error message generated by the schema validation if a mistake is detected
# In your prompts, you should tell LLM:
# - The original wrong output by feeding candidate_json_output
# - The guidance on correction by feeding error_message

def JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
    [YOUR PROMPT HERE]
    """
    auto_repair_prompt = f"""
    [YOUR PROMPT HERE]
    """
    return auto_repair_system_msg, auto_repair_prompt

# ---- Step 2: Run the for loop to go over all the restaurant data in the list (approximately 20 minutes) ----

# You will structure each restaurant description paragraph into a JSON-formatted output by iterating through the dataset using a for loop.
# Although using an LLM to auto repair the response format is not the best practice, here you are guaranteed that the generation and repair tasks here are simple enough for LLMs to accomplish.

### Your Code Here:
structured_restaurant_lists = []
for i, restaurant_paragraph in enumerate(restaurant_list):
    ### 2.1: Produce your initial output


    ### 2.2: Validation and Auto Correction loop on the output (Hint: while loop)
    

    ### 2.3: Append your finalized response to the structured_restaurant_lists

    # A manual progress bar
    if (i+1)%20 == 0:
        print(f'{i+1} out of {len(restaurant_list)} is done')

# A final message to notify the completion
print('ALL DONE!!')

# Take a screenshot of the Python code, clearly showing your implementation and the output from Step 2. Name the screenshot M1L1_structure_for_loop.jpg.

# ---- Step 3: Save the list to a JSON file ------------------------------

# You are almost there! First, print the 50th item in your structured_restaurant_lists.

### Your Code Here:
### Print the 50th item in the structured_restaurant_lists

# Then, save your processed data to your folder by running the following code!

structured_restaurant_lists_json = [json.loads(response) for response in structured_restaurant_lists]

#For each item in the restaurant list, assign it with an itemId to be consistent with the one in the user review data:
for i, response in enumerate(structured_restaurant_lists_json):
    response['itemId'] = 1000001 + i
    structured_restaurant_lists_json[i] = response
    
filename = 'structured_restaurant_data.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(structured_restaurant_lists_json, f, indent=4)

# You can download the saved JSON file to your local folder. Be sure to keep it in a safe location, as you will need it for future assignments.

# ----

# ---- Conclusion --------------------------------------------------------

# You have successfully applied GenAI tools to transform the unstructured text data into a well-structured JSON file!

# ---- Authors -----------------------------------------------------------

# Jianping (Mike) Ye
