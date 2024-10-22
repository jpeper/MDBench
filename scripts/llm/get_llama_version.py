import requests
import os
import json

from argparse import ArgumentParser
from dotenv import load_dotenv
from utils import ENV_PATH

load_dotenv(ENV_PATH)

parser = ArgumentParser()
parser.add_argument(
    "--model", type=str, default="meta/meta-llama-3-70b", help="Model name"
)
args = parser.parse_args()


# The URL for the specific model
url = "https://api.replicate.com/v1/models/{model}".format(model=args.model)

# Headers including the API key for authentication
headers = {
    "Authorization": "Token {token}".format(token=os.environ["REPLICATE_API_TOKEN"]),
    "Content-Type": "application/json"
}

# Make the GET request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    print("Latest version: ", result["latest_version"]["id"])

    # with open("response.json", "w") as f:
    #     json.dump(result, f, indent=4)

else:
    print(f"Request failed with status code {response.status_code}")
    print("Response:", response.text)
