import json
import sys

data = json.load(open('2wiki_stuff/wiki_data/dev.json'))

ex = json.dumps(data[0]['context'])

from generation_utils import llm_infer

output = llm_infer(f"{ex}\n\nFor each document, generate a json list of each fact expressed in the document. Then, turn this into a sparse table for the entire document set, where the rows are the different documents/entities, and the columns include the characteristics. Output these in json format, first with the 'document_facts' list and then a 'sparse_facts_table' field", use_json=True)
import pdb
pdb.set_trace()