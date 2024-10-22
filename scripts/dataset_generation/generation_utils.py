import time
import asyncio
import os
import hashlib
import html
import sys

GLOBAL_OUTPUT = []

from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage

from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def is_json(my_json):
    try:
        json_object = json.loads(my_json)
    except ValueError:
        return False
    return True

def pretty_print_json(json_string):
    """
    Pretty prints a json for html 
    """
    parsed = json.loads(json_string)
    pretty_json = json.dumps(parsed, indent=4, sort_keys=True)
    # Convert newlines to <br> and escape HTML special characters
    return '<br>'.join(html.escape(pretty_json).split('\n'))

def contains_letter(series):
    """Check if a pandas Series contains at least one letter."""
    # Drop NaN values and convert to string to handle all data types
    series = series.dropna().astype(str)
    return series.str.contains(r'[A-Za-z]').any()

def check_dataframe_criteria(df, max_rows, max_cols, k):
    # Check if the number of rows exceeds the maximum allowed
    if len(df) > max_rows:
        print(f"Error: The number of rows ({len(df)}) exceeds the maximum allowed ({max_rows}).")
        return False

    # Check if the number of columns exceeds the maximum allowed
    if len(df.columns) > max_cols:
        print(f"Error: The number of columns ({len(df.columns)}) exceeds the maximum allowed ({max_cols}).")
        return False

    return True

def load_csv(file_path, json=True):
    """
    loads a CSV from file
    by default returns json, or if json==False, returns a df
    """
    import csv
    import pandas as pd

    df = pd.read_csv(file_path, delimiter='#')

    if not check_dataframe_criteria(df, 11, 7, 2):
        print("TABLE DOES NOT MATCH CRITERIA, EXITING")
        sys.exit()

    if json:
        json_list = df.to_dict(orient='records')
        return json_list
    
    else:
        df

def generate_html(data):
    """
    Takes a list of strings + json elements and beautifies it for html display
    """
    html_content = """
<html>
<head>
<title>Example</title>
<style>
pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    max-width: 1500px; /* Limit the width to 1000 pixels */
}
.row {
    border: 1px solid #ddd;
    padding: 10px;
    margin-bottom: 10px;
    background-color: #f9f9f9;
    max-width: 1500px; /* Limit the width of the container div as well */
}
</style>
</head>
<body>
"""
    for item in data:
        title = list(item.keys())[0]
        content = item[title]
        if is_json(content):
            content = f"<pre>{pretty_print_json(content)}</pre>"
        else:
            # Convert newlines to <br> for plain text and escape HTML special characters
            content = '<br>'.join(html.escape(content).split('\n'))
        html_content += f"<div><strong>{title}</strong><p>{content}</p></div>"
    html_content += "</body></html>"
    return html_content

def hash_data(data):
    "hashes the data to generate a uuid for an example"
    return hashlib.sha1(json.dumps(data).encode()).hexdigest()[:10]