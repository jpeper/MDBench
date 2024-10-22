from autogen import UserProxyAgent, Agent
from autogen import GroupChat, GroupChatManager
from autogen.agentchat.contrib.agent_builder import AgentBuilder
import autogen
from models import gpt4o_config
import sys
import os
import time
from langchain_community.chat_models import AzureChatOpenAI
import time
import asyncio
import os
import hashlib
import html
import pdb
from langchain.schema import HumanMessage
from scripts.llm.utils import ENV_PATH
from dotenv import load_dotenv

from llm_infer import llm_infer

import asyncio
import logging
from langchain.schema import HumanMessage

from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

prompt = """
I am trying to assess the difficulty level of the provided evaluation example. I will do this over 5 different types of reasoning characteristics.
Instead of generating an absolute difficulty score, instead, I am providing 2 other candidates, and you will rank the relative difficuly of the final example w/ the other two.
For the provided evaluation example, I want you to score each of the characteristics as {0, 1, 2}, where 0 means this it's easiest of the three, 1 means it's the middle difficulty, or 2 if it is the hardest. To be consistent, I want the output to be json format like this:
{
  "concise_answers": <one one paragraph or less (for each), concisely answer each problem so you can quantify the difficulty,
  "scores": {
        "multi-hop": {'rationale': <justificaton for why you chose this ranking>, 'score': <0|1|2>},
        "temporal": {'rationale': <...>, 'score': <0|1|2>}
        "numeric": ...
        "information_aggregation": ...
        "soft_reasoning": ...
        }
}

[Multi-hop Reasoning] – The ability to solve problems requiring multiple steps to arrive at a solution

Example
Baseline Example: Which country had the most showings and how many was this in total?
Original table: 
|index|date|territory|showings|
|1|october 20, 2006|turkey|200|
|2|october 20, 2006|belgium|600|

Answer: Belgium had the most with 600 showings.
Answer rationale: Turkey had 200 showings and Belgium had 600. 600 > 200, therefore Turkey had the most showings.
Commentary: This is a simple reasoning process as it requires a simple comparison of two values with no additional reasoning required.

Harder Example: Which country had the most showings and how many was this in total?
Edited Table:

|index|date|territory|showings|
|1|october 20, 2006|turkey|200|
|2|october 20, 2006|belgium|600|
|3|october 25, 2006|turkey|500|

Answer: Turkey had the most with 700 showings.
Answer Rationale: Turkey had showings on two different days, so the total is 200+500=700 showings. 700 > Belgium’s 600, therefore Turkey had the most.
Commentary: By adding a new row with complementary information, we necessitate an additional reasoning hop to correctly answer the question. Note that this table was edited specifically such that the answer (Turkey) is flipped from the original answer (Belgium) in the simple example. Edits like these ensure the reasoning cannot be shortcutted (e.g., by simply selecting the row with the highest showings). 

[Temporal Reasoning] – The ability to handle temporal information and dependencies.

Baseline Example: How many total showings were there in each month?

|index|date|territory|showings|
|1|october 20, 2006|turkey|200|
|2|november 21, 2006|belgium|600|
|3|november 21, 2006|turkey|400|
|4|november 22, 2006|belgium|600|

Answer: October 2006 had 200 showings, while November had 1,600
Answer Rationale: October had just one day with 200 showings. November had 3 showings total, summing to 600+400+600 showings total.
Commentary: This is fairly straightforward as we simply sum all rows sharing the same month.

Harder Example: How many total showings were there in each month?

|index|date|territory|showings|notes|
|1|october 20, 2006|turkey|200|Opening day in Turkey|
|2|november 21, 2006|belgium|600|Opening day in Belgium|
|3|the week after opening day|turkey|400||
|4|november 23, 2006|belgium|600||

Answer: October 2006 had 600 showings, while November had 1,200
Answer Rationale: In Turkey, the week after opening day fell in the month of October, therefore there were 200 (from opening day) + 400 (from the week after) = 600 showings in October. November had 600+600 = 1,200 showings, all from Belgium.
Commentary: We introduce a cross-row dependency here that requires temporal reasoning to solve. Namely, we need to intuit that, given opening day is on October 20th, the week immediately following it must fall within the month of October. Again, we intentionally edit the values in the table (and add a ‘notes’ column) to ensure that the answer (600 in October, 1200 in November) necessarily required resolving this cross-row dependency.

[Numeric Reasoning] – The ability to handle numeric values and perform numerical operations
Baseline Example: Rank each day by the total showings.

|index|date|territory|showings|
|1|october 20, 2006|turkey|200|
|2|november 21, 2006|belgium|600|
|3|november 21, 2006|turkey|400|
|4|november 22, 2006|belgium|600|

Answer: November 21st had the most showings with 1000, followed by November 22nd, then October 20th.
Answer Rationale: November 21st had 1000 totals showings – 600 in Belgium and 400 in Turkey. This was greater than the 600 on November 22nd and the 200 on October 20th.
Commentary: This is a simple case of performing numeric operations, having to sum values over different rows to identify the correct answer.

Harder Example: Rank each day by the total sales

|index|date|territory|showings|Avg. sales per showing ($)|
|1|october 20, 2006|turkey|200|6000|
|2|november 21, 2006|belgium|600|1000|
|3|november 21, 2006|turkey|400|1000|
|4|november 22, 2006|belgium|600|500|

Answer: October 20th had the highest sales, followed by November 21st, then November 22nd
Answer Rationale: October 20 had 200 showings * $6000 per showing = $1,200,000. November had 600*$1000 = $600,000 from Belgium and 400*$1000 = $400,000 from Turkey, totalling $1,000,000. November 22 had 600 * $500 = $300,000 in sales.
Commentary: This reasoning requires calculating values over two different columns, and then additionally summing values over associated rows (e.g. the november 21 entries).


[Information Aggregation] – The ability to align, compare and/or contrast knowledge that may be present. This includes non-numeric knowledge.

Baseline Example: Rank the teams by number of wins in the series.

|index|race|pole position|winning team|
|1|May 7, 1992|nico valencia|ferrari|
|2|May 21, 1992|mark steedman|bmw|
|3|June 4, 1992|bonnie bobcat|mclaren|
|4|June 18, 1992|elio muchin|renault|
|5|July 2, 1992|tammy tiger|ford|
|6|July 16, 1992|tyrell eshar|ferrari|
|7|July 30, 1992|alain prost|ferrari|
|8|August 13, 1992|tigre trees|renault|


Answer: Ferrari, Renault, and T-3 are BMW, McLaren and Ford.
Answer Rationale: Ferrari was listed as the winning team three times, Renault twice, and the others once each. 
Commentary: This is a simple example that required calculating the number of appearances of each team in the ‘winning team’ column.

Harder Example: Identify the top two teams in this race series, and explain any correlation between their success and the weather.

|index|race|pole position|winning team|notable conditions|
|1|May 7, 1992|nico valencia|ferrari|sunny + dry|
|2|May 21, 1992|mark steedman|bmw|rainy|
|3|June 4, 1992|bonnie bobcat|mclaren|heavy rain|
|4|June 18, 1992|elio muchin|renault|slick roads|
|5|July 2, 1992|tammy tiger|ford|cold and blustery|
|6|July 16, 1992|tyrell eshar|ferrari|sunny|
|7|July 30, 1992|alain prost|ferrari|overcast|
|8|August 13, 1992|tigre trees|renault|damp|

Answer: Ferrari finished first and Renault finished second. Ferrari’s wins were exclusively in conditions with dry pavement, whereas Renault won only in wet conditions.
Answer Rationale: Ferrari had three wins, and Renault had two wins. The rest of the teams had only one. Notably, Ferrari winning races were only in conditions where the roads were presumably dry (sunny+dry, sunny, and overcast), and Renault’s wins were only on day where the conditions were wet (slick roads, and damp).
Commentary: This answer requires not only understanding the winning teams, but also realizing that there were patterns in the conditions for both teams. Namely, one had to ascertain that Ferrari performed well on dry days, whereas Renault did well on wet roads. This requires aggregating, comparing, and contrasting values across different rows and teams.

[Soft Reasoning] – The ability to reason abductively and make informed decision in cases where some uncertainty may be present.

Simple Example: Who had the most championships?

|index|Year|Championship Winner|
|1|2008|Yusef|
|2|2009|Mattingly|
|3|2010|Tigre Trees|
|4|2011|Yusef “Skeeps” Mattingly|
|5|2012|Tigre|
|6|2013|John Smith|
|7|2014|John Smith|
|8|2015|Harrison Chevrolet|

Answer: Yusef Mattingly, who had wins in 2008, 2009, and 2011
Answer Rationale: Although not clearly stated, some of the entries likely refer to the same person, just sometimes using only the first name, last name, or a nickname. We can reasonably assume ‘Yusef’, ‘Mattingly’, and ‘Yusef “Skeeps” Mattingly’ all refer to the same individual. Similarly, we see both a ‘Tigre Trees’ and ‘Tigre’ which likely refer to the same. 
Commentary: This is an example abductive or ‘best guess’ soft reasoning where one could reasonably assume that some of the entries refere to the same canonical entity/person. Notably, this example is one where a wrong answer would be generated by using a simple exact match heuristic as ‘John Smith’ shows up twice, which is less than Yusef Mattingly.

Harder Example: Rank the countries by total sales.

|index|Country|Sales ($)|Notes|
|1|October 20|Turkey|146200||
|2|October 25|Belgium|39000||
|3|October 25|Germany|134000||
|4|October 26|Austria|42000||
|5|October 26|Netherlands|54000||
|6|October 27|United Kingdom|534700||
|7|October 26|<one that was already mentioned>|195000, roughly 60k more than yesterday's sales.|A follow-up to a prior entry|


Answer: United Kingdom, Germany, Turkey, Netherlands, Austria, Belgium
Answer Rationale: Most country sales are confined to just one row. However, the final row contains sales information that implicitly refers to a country. We see that this country is already mentioned and that this row is a follow-up to a previous entry with sales numbers. The sales value is $195,000 which is stated as 60k more than the prior day sales. We can use this to ascertain what the country is. Namely, we see that there are two entries for the prior day (October 25). Of these two, Germany’s sales were $134,000 which is approximately $60,000 less than $195,000. Belgium’s sales were much lower (over $150k less than $195,000). Therefore, we can reasonably conclude that the October 26 entry in mention refers to Germany. Combining the $134,000 from October 25 and $195,000 from October 26, we see Germany’s total sales are $329,000, which is less than the United Kingdom, but more than Turkey.
Commentary: This problem requires that one notices that the final row can be linked to a prior row. Once this is done, there is some soft reasoning that clearly leads to the proper solution. So, while there is some abduction reasoning required, it is very clear once you put the pieces together.

Here are the 3 examples to compare. Again, generate a score of 0/1/2 for each characteristic, where 0 means the evaluation candidate is the easiest in this category, 1 means middle, 2 means highest.
"""


import os
import json

def load_json_files(folder_path):
    json_data = []
    
    # List all files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            
            # Open and load the JSON file
            with open(file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    json_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {file_name}: {e}")
    
    return json_data

# Specify the folder path where the JSON files are stored
folder_path = '/Users/jpeper697/Documents/mercury_dumps/MultiAgent/MDDG/data/datasets/main/test'

# Call the function to load all JSON files
data_json = load_json_files(folder_path)

# Now json_files_content contains the data of all the loaded JSON files
#print(json_files_content)

prompts = []
num_exs = len(data_json)
import random
random.seed(12345)
random.shuffle(data_json)
for idx, ex in enumerate(data_json):
    num_comparisons = 2
    comparison_indices = [(idx + k) % num_exs for k in [1,2]]

    cand1 = data_json[comparison_indices[0]]
    cand2 = data_json[comparison_indices[1]]
    cand1_ex = f"Comparison 1 Context: {cand1['documents']}\n\n'Comparison 1 Evaluation Question: {cand1['question']}"
    cand2_ex = f"Comparison 1 Context: {cand2['documents']}\n\n'Comparison 1 Evaluation Question: {cand2['question']}"
    example = f"Evaluation Context: {ex['documents']}\n\n'Evaluation Question: {ex['question']}"
    full_prompt = f"{prompt}\n{cand1_ex}\n{cand2_ex}\n{example}\nNow generate the relative scoring for the final context:"
    prompts.append(full_prompt)

import pandas as pd
from io import StringIO
import re
def estimate_tokens(text):
    # Use a regular expression to split the text by spaces and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return len(tokens)


def get_dataset_statistics(data_json):
    total_stats = []
    for ex in data_json:
        try:
            # Use StringIO to simulate reading from a file
            df = pd.read_csv(StringIO(ex['table']), sep="|", engine='python')
            
            # Drop empty columns if there are any (due to leading/trailing separators)
            df = df.dropna(how='all', axis=1)

            num_rows, num_cols = df.shape
            num_cols = num_cols-1
            stats = {
                'num_rows': num_rows,
                'num_cols': num_cols,
                'num_tab_tokens': estimate_tokens(ex['table']),
                'num_doc_tokens': estimate_tokens(ex['documents']),
                'avg_doc_tokens': estimate_tokens(ex['documents']) / num_rows,
            }
            total_stats.append(stats)
        except:
            print("skipping")

    import pdb
    pdb.set_trace()

get_dataset_statistics(data_json)




outputs = llm_infer(prompts)

processed = []

import pdb
pdb.set_trace()

for elt in outputs:
    try:
        processed.append(json.loads(elt))
    except:
        import pdb
        pdb.set_trace()


scores = [{key: val['score'] for key, val in elt['scores'].items()} for elt in processed]

from collections import Counter

keys = set(key for d in scores for key in d.keys())

# Loop through each key and count the frequencies
for key in keys:
    # Collect values for the current key if present in the dictionary
    values = [d[key] for d in scores if key in d]
    
    # Count the frequency of values for the current key
    frequency = Counter(values)
    
    # Print the frequency for this key
    print(f"Frequency of '{key}': {frequency}")
   
output = {ex['example_id']: scores for ex, scores in zip (data_json, scores)}

import pandas as pd

def compress_across_examples(data):
    # Convert input data to a DataFrame
    df = pd.DataFrame(data).T
    
    # Create a function to bin each column (characteristic) across all examples
    def bin_column(col):
        # Count the occurrences of each value (0, 1, 2)
        value_counts = col.value_counts()
        
        # Calculate the sum of frequencies for binning 0+1 and 1+2
        bin_01 = value_counts.get(0, 0) + value_counts.get(1, 0)
        bin_12 = value_counts.get(1, 0) + value_counts.get(2, 0)
        
        # Determine the most balanced binning and apply it
        if abs(bin_01 - value_counts.get(2, 0)) < abs(bin_12 - value_counts.get(0, 0)):
            # Merge 0+1 into 0 and keep 2 as 1
            return col.apply(lambda x: 0 if x < 2 else 1)
        else:
            # Merge 1+2 into 0 and keep 0 as 1
            return col.apply(lambda x: 1 if x > 0 else 0)
    
    # Apply the binning function to each column in the DataFrame
    binned_df = df.apply(bin_column)
    
    return binned_df.T.to_dict()

# Process the input data across examples
output = compress_across_examples(output)


with open('characteristics_dump_1.json', 'w') as ofile:
    json.dump(output, ofile, indent=4)
import pdb
pdb.set_trace()


