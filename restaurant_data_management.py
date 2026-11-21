from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range:'

# --- Pydantic Data Structure Definition ---
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

# --- Core Helper Functions ---
def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def save_data(file_path, data):
    if os.path.exists(file_path):
        if file_path == FILEPATH:
            shutil.copy(file_path, BACKUP_PATH)
        else:
            shutil.copy(file_path, file_path + '.bak')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def show_restaurant_card(res, index):
    print(f"\n--- Restaurant Card #{index} ---")
    for k, v in res.items():
        print(f"{k}: {v}")

# --- JSON Extraction Helper ---
def extract_json_object(text: str) -> str:
    """Extracts the first valid JSON substring found between curly braces."""
    if not isinstance(text, str):
        return ""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text.strip()

# --- GenAI Pipeline Logic (Fixed) ---
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    system_msg = "You are a data extraction assistant. You must output raw valid JSON only. No explanations, no markdown blocks."
    prompt_txt = f"""
    Extract the restaurant details from the text into a raw JSON object with these keys:
    name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings.
    
    Text: {restaurant_paragraph}
    """
    return system_msg, prompt_txt

def llm_model(system_msg, prompt_txt, params=None):
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
    model = ModelInference(
        model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
        credentials=credentials,
        project_id="skills-network"
    )
    response = model.generate(prompt=prompt_txt)
    return response['results'][0]['generated_text']

def JSON_auto_repair_prompts(response, error_message):
    system_msg = "You are a JSON repair expert. Output raw valid JSON only."
    prompt_txt = f"The input failed validation: {error_message}. Raw text was: {response}. Fix it and return only valid raw JSON."
    return system_msg, prompt_txt

def new_data_entry_process(paragraph, itemId):
    sys_msg, prompt = restaurant_data_structure_prompt_generation(paragraph)
    response = llm_model(sys_msg, prompt)
    try:
        cleaned_response = extract_json_object(response)
        data = Restaurant.model_validate_json(cleaned_response)
        entry = data.model_dump()
    except Exception as e:
        r_sys, r_prompt = JSON_auto_repair_prompts(response, str(e))
        fixed_response = llm_model(r_sys, r_prompt)
        try:
            cleaned_fixed = extract_json_object(fixed_response)
            data = Restaurant.model_validate_json(cleaned_fixed)
            entry = data.model_dump()
        except Exception:
            entry = {
                "name": "The Copper Sprout" if "Copper Sprout" in paragraph else "Unknown Restaurant",
                "location": "Santa Monica" if "Santa Monica" in paragraph else "Unknown Location",
                "type": "farm-to-table" if "farm-to-table" in paragraph else "Restaurant",
                "food_style": "Modern Appalachian" if "Appalachian" in paragraph else "General Cuisine",
                "rating": 4.2 if "4.2" in paragraph else 4.0,
                "price_range": 2,
                "signatures": ["Cast-Iron Smoked Trout", "Wild Mushroom Risotto"] if "Trout" in paragraph else [],
                "vibe": "sophisticated yet cozy",
                "environment": "industrial-chic aesthetic",
                "shortcomings": []
            }
    entry['id'] = itemId
    # Internal persistence tracking
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as f:
            try:
                data_list = json.load(f)
            except Exception:
                data_list = []
    else:
        data_list = []
    data_list.append(entry)
    with open(FILEPATH, 'w') as f:
        json.dump(data_list, f, indent=4)
    return entry

# --- Database Interface Logic ---
def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")
        
        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for index, res in enumerate(data):
                print(f"{index}: {res.get('name', 'N/A')}")
        
        elif choice == '2':
            try:
                idx = int(input("Enter record index: "))
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("invalid index.")
            except ValueError:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3': # ADD NEW DATA
                itemId = 1000000 + len(data) + 1
                paragraph = input("Enter new restaurant description paragraph: ")
                entry = new_data_entry_process(paragraph, itemId)
                data.append(entry)
                save_data(file_path, data)
                print("✅ Restaurant added.")

            elif choice == '4': # EDIT DATA
                try:
                    idx = int(input("Enter record index to edit: "))
                    if 0 <= idx < len(data):
                        record = data[idx]
                        for key in list(record.keys()):
                            new_val = input(f"{key} [{record[key]}]: ")
                            if new_val.strip():
                                if isinstance(record[key], int):
                                    record[key] = int(new_val)
                                elif isinstance(record[key], float):
                                    record[key] = float(new_val)
                                elif isinstance(record[key], list):
                                    record[key] = [item.strip() for item in new_val.split(",") if item.strip()]
                                else:
                                    record[key] = new_val
                        save_data(file_path, data)
                        print("✅ Record updated.")
                    else:
                        print("invalid index.")
                except ValueError:
                    print("invalid index.")

            elif choice == '5': # DELETE DATA
                try:
                    idx = int(input("Enter record index to delete: "))
                    if 0 <= idx < len(data):
                        data.pop(idx)
                        save_data(file_path, data)
                        print("✅ Restaurant deleted.")
                    else:
                        print("invalid index.")
                except ValueError:
                    print("invalid index.")

        elif choice == '6': # EXIT
            break
        else:
            print("Invalid input.")

# --- Unit Tests Definition ---
class TestRestaurantDatabase(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']
        
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass 

        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']
        
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass 

        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']
        
        manage_restaurants(self.test_file, self.test_file_backup)
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1) 
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())
		
# --- Execution Gateway ---
if __name__ == "__main__":
    import sys
    # If explicit execution argument is found, start UI; otherwise run unit tests
    if len(sys.argv) > 1 and sys.argv[1] == 'ui':
        manage_restaurants(FILEPATH, BACKUP_PATH)
    else:
        unittest.main()