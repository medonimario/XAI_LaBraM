from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Access the variable
dataset_path = os.getenv("DATASET_PATH")

print("Dataset path is:", dataset_path)