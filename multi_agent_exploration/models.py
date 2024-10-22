# add imports
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

from openai import AzureOpenAI, OpenAI
import os

model_name = "gpt-4o"
gpt4o_config = {"model": model_name,
                "api_key": os.getenv("OPENAI_BASE_URL"),
                "api_type": "azure",
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "api_version": "2024-02-01",
                "temperature": 0.0,}

class Model:
    def __init__(self, model_name, temperature=0.0, max_tokens: int =256):
        self.model_name = model_name
        api_key = os.getenv("OPENAI_BASE_URL")
        endpoint = os.getenv("OPENAI_BASE_URL")
        if "openai" in endpoint:
            self.client = AzureOpenAI(api_key=api_key,
                                      azure_endpoint=endpoint,
                                      api_version="2024-02-01")
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=endpoint,
            )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).choices[0].message.content
        return response