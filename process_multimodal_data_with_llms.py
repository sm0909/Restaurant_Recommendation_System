"""
Process Multimodal Data with LLMs

Standalone Python version of the original analysis notebook.

Dependencies:
    pip install ibm-watsonx-ai==1.4.7 matplotlib==3.10.7 numpy==2.3.4

Configuration (environment variables, no values hardcoded):
    RECIPES_URL  - source URL for Recipes.json
    SYNTHETIC_USER_REVIEWS_URL  - source URL for Synthetic-User-Reviews.json
    SYNTHETIC_RECIPE_IMAGES_URL  - source URL for synthetic-recipe-images.zip
    LLM_MODEL_ID  - model identifier for your LLM provider
    LLM_PROJECT_ID  - project / workspace id for your LLM provider
    LLM_API_URL  - base endpoint URL for your LLM provider
"""

import os
import shutil
import tarfile
import urllib.request
import zipfile

# ---- Configuration -------------------------------------------------
# Values are read from the environment so nothing is hardcoded.
RECIPES_URL = os.environ.get("RECIPES_URL", "")  # source URL for Recipes.json
SYNTHETIC_USER_REVIEWS_URL = os.environ.get("SYNTHETIC_USER_REVIEWS_URL", "")  # source URL for Synthetic-User-Reviews.json
SYNTHETIC_RECIPE_IMAGES_URL = os.environ.get("SYNTHETIC_RECIPE_IMAGES_URL", "")  # source URL for synthetic-recipe-images.zip
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "")  # model identifier for your LLM provider
LLM_PROJECT_ID = os.environ.get("LLM_PROJECT_ID", "")  # project / workspace id for your LLM provider
LLM_API_URL = os.environ.get("LLM_API_URL", "")  # base endpoint URL for your LLM provider


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


def display(obj):
    """Console stand-in for the notebook `display` helper."""
    show = getattr(obj, "show", None)
    if callable(show):
        show()
    else:
        print(obj)

# ---- Lab: Process Multimodal Data with LLMs ----------------------------

# ---- Scenario ----------------------------------------------------------

# ---- Background --------------------------------------------------------
# You transformed unstructured restaurant descriptions into a structured JSON knowledge base using LLMs. This allowed the app to reason over textual attributes such as cuisine type, pricing, and signature dishes in a consistent, machine-readable format.
# However, the app’s data is not limited to text. It also includes food recipes and user visit histories that reference images, capturing visual information that text alone cannot fully describe, such as presentation style, ingredients, and portion details. To unlock this information and integrate it with the original databases, you will use multimodal GenAI capabilities to convert images into descriptive text and enrich the structured knowledge you have.
# ---- The challenge -----------------------------------------------------
# Images are inherently unstructured and cannot be directly stored or queried in a text-based knowledge system. To make them usable, you must first generate high-quality textual descriptions that summarize the visual content of each image.
# Your challenge in this lab is to:
# Apply image captioning to generate descriptive text for food images
# Align and merge image captions with the corresponding recipe and restaurant JSON records
# Extend the existing structured knowledge base with multimodal context while preserving a consistent schema
# By the end of this lab, you will have enriched your restaurant knowledge base with visual insights. You will bridge text and images into a unified, machine-accessible representation, setting the stage for more intelligent multimodal reasoning in later assignments.

# ---- Objectives --------------------------------------------------------
# In this lab, you will write a Python program that will:
# Load food recipe and user visit data containing image references
# Use a multimodal foundation model to generate textual captions for food images
# Integrate the generated image captions into the existing structured JSON data
# Produce an enriched multimodal JSON knowledge file for downstream applications

# ---- Important: About the lab environment ------------------------------

# Please be aware that sessions for this lab environment are not persisted. Every time you connect to this lab, a new environment is created for you. Any data you may have saved in the earlier session would get lost. Plan to complete these labs in a single session, to avoid losing your data.

# ---- Screenshot requirement for this lab -------------------------------

# You will be prompted to take a screenshot and save it on your own device. You will need this screenshot either to answer graded quiz questions or to upload as your submission for the Final Project at the end of this course. You can use various free screen-grabbing tools or your operating system's shortcut keys to do this (for example, Alt+PrintScreen on Windows and Command+shift+4 on Mac).
# Note: The screenshot can be saved with either the .jpg or .png extension.

# ----

# ---- Set up the lab environment ----------------------------------------

# For this lab, you will still be using the following libraries:
# numpy for numerical operations and handling array-based data during preprocessing and analysis
# matplotlib for basic data visualization and plotting, useful for inspecting distributions or intermediate results
# json for parsing, constructing, and serializing structured JSON representations extracted from unstructured text
# Credentials to securely authenticate with the watsonx.ai service
# ModelInference to invoke foundation models for text understanding and information extraction
# GenTextParamsMetaNames to configure text generation and extraction parameters
# ModelTypes and DecodingMethods to select the appropriate model and control inference behavior
# These libraries together enable you to transform images into text-based data using GenAI-powered workflows.

# ---- Installing required libraries -------------------------------------
# Run the following code block to install all required libraries:

# ---- Importing required libraries --------------------------------------
# It is recommended to import all required libraries in one place (here):

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

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
# Run the following code to fetch the food recipe, user visit history, and recipe images.

download_file(RECIPES_URL, "Recipes.json")
download_file(SYNTHETIC_USER_REVIEWS_URL, "Synthetic-User-Reviews.json")
print(download_file(SYNTHETIC_RECIPE_IMAGES_URL, "synthetic-recipe-images.zip"))

# Run the following code to unzip the image folder:

import zipfile

with zipfile.ZipFile("synthetic-recipe-images.zip", 'r') as zip_ref:
    zip_ref.extractall()

# ----

# ---- Exercise 1: Preprocess the food reciepe data ----------------------

# ---- Step 1: Explore the food recipe JSON File and its images ----------
# Before applying any multimodal processing, it is important to understand the structure and contents of the data you will be working with. In this step, you will load the food recipe JSON file and examine its fields, including textual metadata and image references.
# You will also inspect a sample of the associated food images to understand the visual information they contain, such as ingredients, presentation style, and portion size. This exploration step helps you identify how the image data aligns with recipe metadata and prepares you for integrating image captions into the existing JSON structure in later steps.
# By the end of this step, you should have a clear understanding of how the recipe data and images are organized, and how they relate to the structured restaurant knowledge built in the previous lesson.

import glob
# Robust positional fallback for image files extracted from the zip:
SORTED_IMAGES = sorted(
    glob.glob("**/*.png", recursive=True) + glob.glob("**/*.jpg", recursive=True)
)

def get_recipe_image_path(index, recipe):
    """Return a usable image path for a recipe, whether it stores a field or aligns by order."""
    for key in ("image_path", "image", "image_file", "img", "filename"):
        if isinstance(recipe, dict) and recipe.get(key):
            cand = recipe[key]
            if os.path.exists(cand):
                return cand
            hits = glob.glob(f"**/{os.path.basename(cand)}", recursive=True)
            if hits:
                return hits[0]
    return SORTED_IMAGES[index]

# ===== Exercise 1 — Step 1: Explore recipes + show first image =====
### 1.1: Load the json file
with open("Recipes.json", "r", encoding="utf-8") as f:
    recipe_data = json.load(f)

### 1.2: Print each key-value pair of the first recipe -> key (type of value): value
for key, value in recipe_data[0].items():
    print(f"{key} ({type(value).__name__}): {value}")

### 1.3: Show the image of the first recipe (recipe1)
first_image_path = get_recipe_image_path(0, recipe_data[0])
print(f"\nShowing image: {first_image_path}")
print(display(Image.open(first_image_path)))

# ---- Step 2: Define the vision LLM with LLaMA --------------------------
# To generate meaningful textual descriptions from food images, you will use a vision-capable large language model (LLM). In this step, you will define and configure a LLaMA-based vision model that can process both images and text prompts.
# You will initialize the model with the appropriate credentials and inference parameters, and verify that it accepts image inputs and produces descriptive captions. This model will serve as the core component for converting visual information into natural language that can be integrated with your existing JSON-based knowledge structure.
# By completing this step, you will have a vision-enabled LLM ready to perform image captioning reliably and repeatably.

# ===== Exercise 1 — Step 2: Define the vision LLM (LLaMA) =====
import base64

def vision_llm(system_msg, prompt_txt, image_path):
    model_id = LLM_MODEL_ID
    project_id = LLM_PROJECT_ID
    credentials = Credentials(url=LLM_API_URL)
    generate_params = {"max_tokens": 300}

    ### 2.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=generate_params,
    )

    ### 2.2: Encode the input image to a base64 string
    with open(image_path, "rb") as img_f:
        encoded_image = base64.b64encode(img_f.read()).decode("utf-8")

    ### 2.3: Define the messages for the model
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_txt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
            ],
        },
    ]

    ### 2.4: Get the response for the messages
    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]

# ---- Step 3: Design and validate prompts for the vision LLM ------------
# To ensure the vision LLM generates useful and consistent image captions, careful prompt design is essential. In this step, you will create a prompt that instructs the model to focus on relevant visual details, such as ingredients, cooking style, and presentation, while avoiding unnecessary or speculative information.
# Once the prompt is defined, you will perform a simple unit test by running the model on a small sample of food images. This allows you to verify that the generated captions are accurate, concise, and suitable for integration into the existing JSON data structure.
# By the end of this step, you will have a validated prompt that reliably converts food images into high-quality textual descriptions.

# ===== Exercise 1 — Step 3: Caption prompt template + unit test =====
def image_caption_prompt_template(food_name):
    image_caption_system_msg = (
        "You are a professional food photographer and culinary writer. You write concise, "
        "accurate captions describing only what is visible in a food photo: the dish, key "
        "ingredients, cooking/plating style, garnishes, and presentation. Be factual and "
        "avoid speculation or marketing language."
    )
    image_caption_prompt_txt = (
        f'The attached image is a photo of "{food_name}". Write a concise 2-3 sentence '
        f'caption describing the dish shown, focusing on visible ingredients, preparation '
        f'style, and presentation. Keep "{food_name}" as the focus.'
    )
    return image_caption_system_msg, image_caption_prompt_txt

### 3.2: Get the prompts with the food name of the first recipe
first_food_name = recipe_data[0].get("name", "recipe 1")
sys_msg, prompt_txt = image_caption_prompt_template(first_food_name)

### 3.3: Get the test response and print it
response = vision_llm(sys_msg, prompt_txt, get_recipe_image_path(0, recipe_data[0]))
print(response)

# ---- Step 4: Caption all images and augment the data -------------------
# With a validated prompt and a working vision LLM, you will now scale the image captioning process across the entire dataset. In this step, you will iterate through all food images, generate captions for each one, and associate the results with their corresponding recipe or restaurant records.
# You will then augment the existing JSON data by adding the generated image captions as new fields, ensuring the final structure remains consistent and machine-readable. This enriched dataset combines textual and visual insights into a unified representation, extending the structured knowledge base created in the previous lessons.
# By the end of this step, you will have a multimodal JSON file that integrates image-derived descriptions with existing structured data, ready for downstream GenAI applications.

# ===== Exercise 1 — Step 4: Caption ALL recipe images (up to ~15 min) =====
for i in range(len(recipe_data)):
    if (i + 1) % 20 == 0:
        print(f'{i+1} out of {len(recipe_data)} is done')
    ### Step 4.1: Get the caption prompts
    food_name = recipe_data[i].get("name", f"recipe {i+1}")
    sys_msg, prompt_txt = image_caption_prompt_template(food_name)
    ### Step 4.2: Get the response with the prompts
    image_path = get_recipe_image_path(i, recipe_data[i])
    response = vision_llm(sys_msg, prompt_txt, image_path)
    recipe_data[i]['image_description'] = response
print('ALL DONE!')
# Save (provided)
filename = 'augmented_food_recipe.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(recipe_data, f, indent=4)

# Take a screenshot of the Python code, clearly showing your implementation and the output from Step 2. Name the screenshot M1L2_caption_all_recipes.jpg.

# ---- Step 5: Save the image-caption-augmented recipe data --------------
# Save your data by running the following code.

filename = 'augmented_food_recipe.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(recipe_data, f, indent=4)

# You can download the saved JSON file to your local folder. Be sure to store it in a safe location, as you will need it for future assignments.

# ----

# ---- Exercise 2: Preprocess the user visit history ---------------------
# In this exercise, you will largely replicate the workflow from Exercise 1. The core steps, data exploration, prompt design, model inference, and JSON augmentation, remain the same.
# The key difference lies in prompt design. Instead of focusing on food images, you will tailor your prompt to generate concise, informative descriptions from user visit history data, such as contextual cues from URLs or associated images combined with their written reviews. These prompts should emphasize elements that help explain user preferences and behavior while maintaining consistency with the existing JSON schema.
# By the end of this exercise, you will have enriched the user visit history data with structured, machine-readable descriptions, further expanding the multimodal knowledge base built throughout this lab.

# ---- Step 1: Load the user review data ---------------------------------
# Similar to Step 1 of Exercise 1, complete the following code block to load and explore the user review data.

# ===== Exercise 2 — Step 1 (Part 1): Load user reviews =====
### 1.1: Load review dataset
with open("Synthetic-User-Reviews.json", "r", encoding="utf-8") as f:
    user_review_data = json.load(f)

### 1.2: Print the first review by key-value pairs
for key, value in user_review_data[0].items():
    print(f"{key} ({type(value).__name__}): {value}")

# Let's look at the 'images' item. Although it looks like a list, it is actually a list in string type. Therefore, you need to be careful and convert the string to the actual Python list. Complete the code block below to display the first review image of the first restaurant visit.

# ===== Exercise 2 — Step 1 (Part 2): Convert string-list of images and show first =====
import ast
import requests

### 1.3: Convert the list-of-images (stored as a string) to an actual Python list
review_images = ast.literal_eval(user_review_data[0]['images'])

### 1.4: Get the image content from the first URL
image_data = requests.get(review_images[0])

### 1.5: Write the content to a temporary file
with open('review_image_placeholder.jpg', 'wb') as img_file:
    img_file.write(image_data.content)

### 1.6: Open and show the image
print(display(Image.open('review_image_placeholder.jpg')))

# ---- Step 2: Define the prompt and the validation ----------------------
# Again, you will design a prompt template for the vision task and perform a unit test to validate the model’s behavior. However, unlike the previous exercise, this prompt must incorporate textual context from user reviews to guide the image captioning process.
# In this step, you will define a prompt template function that takes user review text as input and uses it as contextual information when describing a food image. The system message assigns the model the role of a culinary expert, while the prompt instructs the model to generate a concise image description that aligns with the sentiment and details expressed in the reviews.
# After defining the prompt template, you will perform a unit test by running the vision LLM on a single image and its associated reviews. This test allows you to verify that the generated caption effectively combines visual information from the image with contextual cues from the reviews before applying the prompt at scale in later steps.

# ===== Exercise 2 — Step 2: Review-context caption prompt + unit test =====
def review_context_image_caption_prompt_template(reviews):
    review_context_image_caption_system_msg = (
        "You are a culinary expert and food writer. You are given a photo from a restaurant "
        "visit together with the written reviews of that visit. Describe what the image shows "
        "concisely and factually, using the reviews only as supporting context so your "
        "description matches the sentiment and details the reviewers mention. Focus on the "
        "food, presentation, and dining setting; do not invent details."
    )
    review_context_image_caption_prompt_txt = (
        f"Here are the user reviews for this visit:\n{reviews}\n\n"
        f"Using these reviews as context, write a concise 2-3 sentence description of the "
        f"attached image."
    )
    return review_context_image_caption_system_msg, review_context_image_caption_prompt_txt

# helper to read the review text whatever the field is called
def get_review_text(record):
    for key in ("reviews", "review", "review_text", "reviewText", "text"):
        if record.get(key):
            return record[key]
    return ""

### 2.2: Get the prompts (use the first user's reviews)
sys_msg, prompt_txt = review_context_image_caption_prompt_template(get_review_text(user_review_data[0]))

### 2.3: Get the response with vision_llm (on the placeholder image saved in Step 1)
response = vision_llm(sys_msg, prompt_txt, 'review_image_placeholder.jpg')
print(response)

# ---- Step 3: Caption all images and augment the data -------------------
# Now, you are ready to get all captions with a for loop! Complete and run the following code block to caption all the review images.

# ===== Exercise 2 — Step 3: Caption ALL review images =====
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data_with_retry(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response

for i in range(len(user_review_data)):
    ### Step 3.1: Convert the string to the Python list of image URLs
    review_images = ast.literal_eval(user_review_data[i]['images'])

    review_image_captions = []
    if len(review_images) > 0:
        for img_url in review_images:
            try:
                ### Step 3.2: Get image data with retry
                image_data = get_data_with_retry(img_url)
                print("Success!")
            except Exception as e:
                print(f"All retries failed at url {img_url}:", e)
                continue
            image = image_data.content
            with open('review_image_placeholder.jpg', 'wb') as img_file:
                img_file.write(image)

            ### Step 3.3: Get prompts, get response, append
            sys_msg, prompt_txt = review_context_image_caption_prompt_template(get_review_text(user_review_data[i]))
            caption = vision_llm(sys_msg, prompt_txt, 'review_image_placeholder.jpg')
            review_image_captions.append(caption)

    user_review_data[i]['image_captions'] = review_image_captions

print('ALL DONE!')

# Save (provided)
filename = 'augmented_user_review.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(user_review_data, f, indent=4)

# ---- Step 4: Save the image-caption-augmented user review data ---------

# Run the following code block to save the augmented review data:

filename = 'augmented_user_review.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(user_review_data, f, indent=4)

# You can download the saved JSON file to your local folder. Be sure to store it in a safe location, as you will need it for future assignments.

# ----

# ---- Conclusion --------------------------------------------------------
# You have successfully applied GenAI tools to transform the unstructured text data into a well-structured JSON file!
# ----

# ---- Authors -----------------------------------------------------------

# Jianping (Mike) Ye
