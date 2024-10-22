### Multi-step process for sourcing table + iterating
import sys
from MultiAgent.MDBench.scripts.dataset_generation.generation_utils import llm_infer, load_csv
import json
import pandas as pd
from io import StringIO
from llm_infer_fn import send_eval_query
from box import Box
from pathlib import Path
import random

from fuzzy_json import loads as floads

LOG_OUTPUTS = {}

import hashlib

# load prompt templates from config file
config = Box(json.load(open("data_generation_config/generation_config/prompts.json")))
random_seed = sys.argv[3]
random.seed(random_seed)


def dataframe_to_pipe_separated(df):

    # Add index column (zero-based)
    df.index = df.index + 1  # Start index from 1
    df.reset_index(inplace=True)
    df.rename(columns={"index": "index"}, inplace=True)
    # Convert DataFrame to string with pipes
    df_str = df.to_csv(sep='|', index=False)
    
    # Add pipes on the outside of the table
    lines = df_str.split('\n')
    header = '|' + lines[0] + '|'
    rows = ['|' + line + '|' for line in lines[1:] if line]
    
    # Join header and rows
    return '\n'.join([header] + rows)

def xsv_table(json_table, delimiter="|", markdown=False, indexed=False):
    """
    returns a json table as a text table, or optionally as markdown table
    """
    df = pd.DataFrame(json_table)
    if indexed == True:
        return dataframe_to_pipe_separated(df)
    if markdown==True:
        return df.to_markdown()
   
    output_io = StringIO()
    # Export the DataFrame to the StringIO object as TSV
    df.to_csv(output_io, sep=delimiter, index=False)

    # Get the TSV format string from the StringIO object
    tsv_output = output_io.getvalue()
    # Close the StringIO object
    output_io.close()

    return tsv_output

def xsv_to_json(data: str, delimiter='|') -> pd.DataFrame:
    """
    Load |-separated data into a pandas DataFrame, auto-detecting if extra stripping is needed.
    
    Parameters:
        data (str): The |-separated data as a string.
        
    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    # Check if the data needs to be stripped of leading and trailing pipes
    lines = data.strip().split('\n')
    if all(line.startswith(delimiter) and line.endswith(delimiter) for line in lines):
        cleaned_data = '\n'.join([line.strip(delimiter) for line in lines])
    else:
        cleaned_data = data

    # Use StringIO to create a DataFrame
    df = pd.read_csv(StringIO(cleaned_data), delimiter=delimiter)
    
    return df

def shuffle_with_indices(input_list):
    # Create a list of tuples (element, original_index)
    indexed_list = list(enumerate(input_list))
    
    # Shuffle the indexed list
    random.shuffle(indexed_list)
    
    # Extract the shuffled elements and their original indices
    shuffled_list = [element for index, element in indexed_list]
    shuffled_indices = [index for index, element in indexed_list]
    
    return shuffled_list, shuffled_indices

def hash_string(input_string):
    # Create a new hashlib object for SHA-256
    sha256 = hashlib.sha256()
    
    # Encode the input string to bytes
    encoded_input = input_string.encode('utf-8')
    
    # Update the SHA-256 object with the encoded input
    sha256.update(encoded_input)
    
    # Get the hash digest
    hash_bytes = sha256.digest()
    
    # Convert the hash digest to an integer
    hash_int = int.from_bytes(hash_bytes, 'big')
    
    return hash_int

def generate_document_set(table, table_summary, question, answer, config):
    """
    table should be
    """
    random.seed(config.doc_generation.random_seed + hash_string(table))
    gen_template = config.doc_generation.contextual_outline

    rendered_templates = [gen_template.format(table=table, question=question, answer=answer, doc_row=idx+1, description=table_summary, doc_len=random.randint(1, 5)) for idx in range(len(xsv_to_json(table)))]
    
    template_ordering = list(range(len(rendered_templates)))
    if config.doc_generation.shuffle_docs == True:
        rendered_templates, template_ordering = shuffle_with_indices(rendered_templates)
    
    print(template_ordering)
    if len(rendered_templates) > 20:
        exit
        
    result = llm_infer(rendered_templates, use_json=True)
    documents = [floads(elt)['resultant_document'] for elt in result]

    zipped_list = [{'prompt': prompt, 'result': result} for prompt, result in zip(rendered_templates, documents)]
    final_output = "\n\n".join([f"Document {idx+1}: {elt}" for idx, elt in enumerate(documents)])

    output = {"intermediate_results": zipped_list, "final_ex": final_output}

    return output

if __name__ == "__main__":

    # input tsv file as json
    json_input_table = load_csv(sys.argv[1])
    output_folder = sys.argv[2]


    # convert to json format
    tsv_data = xsv_table(json_input_table, "|")
    LOG_OUTPUTS['input_table'] = {'content': json_input_table, 'format': 'xsv'}
    table_summary_prompt = "Generate a short one sentence table name for the following table"
    table_summary_prompt = config.table_loading.table_summary_prompt.format(seed_table=tsv_data)

    input_table_summary = llm_infer(table_summary_prompt)

    LOG_OUTPUTS['input_table_summary'] = {'content': input_table_summary, 'format': 'str'}

    new_tsv_data = xsv_table(json_input_table, indexed=True)

    random.seed(hash_string(new_tsv_data) + int(random_seed))

    reasoning_types = ['multi-hop', 'temporal', 'numeric', 'information_aggregation', 'soft_reasoning']

    # Randomly choose the number of elements to select (between 1 and 5)
    num_elements = random.randint(1, len(reasoning_types))
    selected_elements = random.sample(reasoning_types, num_elements)
    formatted_output = ", ".join([f"{i+1}) {selected_elements[i]}" for i in range(num_elements)]) + "."
    table_edit_prompt = config.table_editing.mdb_table_edit

    rendered_edit_prompt = table_edit_prompt.format(edit_instruction=open(config.table_editing.mdb_edit_demonstration).read(), original_table=new_tsv_data, reasoning_skills=formatted_output)

    output = floads(llm_infer(rendered_edit_prompt, use_json=True))

    mdb_answer = output['final_answer']
    mdb_question = output['final_question']
    mdb_table = output['final_table']
    mdb_edit_plan = output['intermediate_steps']
    mdb_difficulty = output['example_difficulty']
    mdb_difficulty_rationale = output['example_difficulty_rationale']

    new_answer = mdb_answer
    edited_table = mdb_table
    new_edit_plan = mdb_edit_plan
    new_question = mdb_question


    table_validity_rationale_prompt = config.table_editing.edit_answerability_verification
    rendered_validity = table_validity_rationale_prompt.format(seed_table_summary=input_table_summary, seed_table=tsv_data, actual_edits=json.dumps(new_edit_plan), final_table=edited_table,final_question=new_question, final_answer=new_answer)

    validity_rationale = llm_infer(rendered_validity)

    validity_scalar = floads(llm_infer(f"How consistent/valid is this reasoning in the following process for generating an example from a table? Score the validity and consistency of the resultant table+question+answer on a scale of 0-5. I want to be able to identify and ignore examples with low scores that I shouldn't include in my dataset. Note that we actually want the data to be tricky to reason about, so long as there's a valid reasoning path. Output as a json with 'score' and 'brief_rationale' fields. Here is the example: {validity_rationale}", use_json=True))


    table_example_verification_prompt = config.table_example_verification.potential_answers_prompt.format(table=edited_table, question=new_question)
    example_verification = llm_infer(table_example_verification_prompt)

    ambiguity_prompt = config.table_example_verification.ambiguity_prompt.format(question=new_question, solutions=example_verification)
    example_verification_summary = floads(llm_infer(ambiguity_prompt, use_json=True))

        
    doc_gen_output = generate_document_set(edited_table, input_table_summary, new_question, new_answer, config)


    llm_table_answer = send_eval_query(edited_table, new_question, new_answer)

    final_doc = doc_gen_output['final_ex']
    intermediate =  doc_gen_output['intermediate_results']

    eval_results = send_eval_query(final_doc, new_question, new_answer)

    #########################################################################################################

    new_table_markdown = xsv_table(xsv_to_json(edited_table), markdown=True)


    output = {
    "Existing Table Info": xsv_table(xsv_to_json(tsv_data), markdown=True),
    "Generated Edits Rationale": json.dumps(mdb_edit_plan),
    "Resultant Generated Table Markdown": new_table_markdown,
    "Resultant Generated Table -- Bar-separated": mdb_table,
    "Resultant Generated Question": new_question,
    "Resultant Generated Answer": new_answer,
    "Example Difficulty": json.dumps(output['example_difficulty']),
    "Validity Scalar": validity_scalar['score'],
    "Encouraged Characteristics": selected_elements,
    "Validity Rationale": validity_scalar['brief_rationale'],
    "Ambiguity Scalar": example_verification_summary['score'],
    "Ambiguity Rationale": example_verification_summary['rationale'],
    "Example Difficulty Rationale": json.dumps(output['example_difficulty_rationale']),
    "Document Set Intermediate Calls": json.dumps(intermediate, indent=4),
    "Generated Document Set": final_doc,
    "TABLE Reasoning Results -- Ground Truth Answer": llm_table_answer['correct_answer'],
    "TABLE Reasoning Results -- GPT-4 Predicted Answer": llm_table_answer['predicted_answer'],
    "TABLE Reasoning Results -- GPTEval Prediction Evaluation": llm_table_answer['gpt_eval'],
    "TABLE Reasoning Results -- GPTEval Scalar Score": llm_table_answer['accuracy'],
    "DOCUMENT Reasoning Results -- Ground Truth Answer": eval_results['correct_answer'],
    "DOCUMENT Reasoning Results -- GPT-4 Predicted Answer": eval_results['predicted_answer'],
    "DOCUMENT Reasoning Results -- GPTEval Prediction Evaluation ": eval_results['gpt_eval'],
    "DOCUMENT Reasoning Results -- GPTEval Scalar Score": eval_results['accuracy'],
    "output_version": "v4"
    }

    str_output = "\n".join([f"**{key}**\n\n{elt}" for key, elt in output.items()])

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    with open(output_folder + "/full_log.txt", "w") as text_file:
        text_file.write(str_output)

    with open(output_folder + "/full_log.json", "w") as text_file:
        json.dump(output, text_file, indent=2)

