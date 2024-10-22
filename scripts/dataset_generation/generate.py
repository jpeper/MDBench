"Deprecated code from other prompting exploration"
import os

import json
import hashlib
import html
import argparse

import json

from MultiAgent.MDBench.scripts.dataset_generation.generation_utils import generate_html, hash_data, llm_infer

GLOBAL_OUTPUT = []

def log_output(name, value, verbose=True):
    GLOBAL_OUTPUT.append({name: value})
    if verbose:
        print(f"{name}:\n{value}")

def generate_skill_specific_instructions(skill_description, complexity_description, domain_description):
    og_inst = "I want to create a multi-step multi-document reasoning example. The example should necesstitate strong reasoning capacity for a certain skill in order to be solved correctly," \
              "and should also appear natural and have a practical application/scenario/use case." \
              "The example should consist of a (generated) multi-document input, a prompt (ie, a question / instruction), and then a ground truth output.\n" 
    instruction = f"This example should center around proving out the following skill: {skill_description}.\n" \
                  f"It should also be subject to the following considerations.\n" \
                  f"Complexity: {complexity_description}\n" \
                  f"Domain / Use Case Description: {domain_description}\n" \
                  "Output it as a JSON dictionary containing the input documents, instruction, and ground_truth fields. The question or questions in the instruction should be exact-matchable " \
                  "(e.g. true/false, or a short concrete numeric or string value that is unambiguous). the ground truth should be a dictionary of the answers... e.g. {'answer_1': <answer> ...}. there should also be a 'ground_truth_full_explanation' field that contains the justification."
    return og_inst + "\n" + instruction

def introspect_on_problem(problem, i):
    
    introspect_prompt =  f"This a synthetic example for testing the reasoning capabilities of a model. What are a few ways (3-5) of making this example both more realistic and to make the reasoning more nuanced/challenging? It's okay to change the topic / facts if needed. I also want to make the example longer if the documents don't seem full-fledged. Please output a string list of a few improvements that could be made. The example:\n {problem}"
    log_output(f'Introspection Prompt {i}', introspect_prompt)
    introspection = llm_infer(introspect_prompt)
    log_output(f'Introspection Output {i}', introspection)
    return introspection

def refine_problem(problem, i):
    introspection = introspect_on_problem(problem, i)
    refinement_prompt = f"Here is an synthetic data example that I am trying to improve: {problem}\n" \
                        f"Use the following feedback to refine the above problem:\n{introspection}\n" \
                        f"The output of this should be the EXACT original json format that includes the instruction, ground truth and input documents. The output MUST be json-parseable. The question or questions in the instruction should be exact-matchable."
    log_output(f'Refinement Prompt {i}', refinement_prompt)
    refined_problem = llm_infer(refinement_prompt)
    log_output(f'Refinement Output {i}', refined_problem)
    return refined_problem

def send_eval_query(context, input_question):
    og_prompt = f"Input Context:\n{context}\n" \
             f"Question:\n{input_question}.\n"
    

    prompt = og_prompt + "Output it as a JSON dictionary containing the predicted answer(s) to the question(s). The answer or answers in the instruction should be exact-matchable (e.g. true/false, or a short concrete numeric or string value that is unambiguous). The 'answer' field should be a dictionary of the answers... e.g. {'answer_1': <answer> ...}."
    cot_prompt = og_prompt + "Output a JSON dictionary containing the predicted answer(s) to the question(s), and also explanation field. e.g. {'answer': {'answer 1': 'blah'}, 'explanation': '<explanation string>}. The answer or answers should be exact-matchable (e.g. true/false, or a short concrete numeric or string value that is unambiguous)."
     # there should also be a 'ground_truth_full_explanation' field that contains the justification"
    log_output(f'EVAL INPUT', input_question)


    result = llm_infer(prompt)
    cot_result = llm_infer(cot_prompt)

    result = json.loads(result)
    cot_result = json.loads(cot_result)

    result['explanation'] = cot_result['explanation']
    result['cot_answer'] = cot_result['answer']

    log_output('EVAL OUTPUT', json.dumps(result))
    return result

def expand_skills_config(config):
    """
    Takes a json skills config and expands into a list of skills or skill combinations
    """
    skills_list = []
    
    skill_config = config.skill_config
    for group_name, skill_group in skill_config['skill_groups'].items():
        for skill in skill_group:
            skills_list.append(json.load(open(os.path.join(skill_config['base_path'], group_name, skill))))

    return skills_list

def generate_examples(config):
    # Define skills, complexity levels, and domains

    skills_list = expand_skills_config(config)


    skill_descriptions = [elt['description'] for elt in skills_list]

    complexity_levels = [
                         'High. The example should have a high amount of challenge and assume high domain familiarity (using common vernacular and colloqualisms)',
                         'Moderate. The example should have a moderate difficulty for the described skill',
                         'Expert. The example should be an expert-level problem requiring power reasoning capacity and should be situated within the domain in a natural way.'
                        ]

    domains = ['News, news articles, or journalstic perspectives',
               'Scientific content, articles, papers, studies, and content.',
               'Historical settings or topics, perhaps relating to events or historical debates and conflicts',
               'Business-related topics relating to companies, work-environment situations, workplace interactions, money, finance',
               'Consumer opinions, product reviews/comparison, interviews, etc..']

    counter = 0
    for skill in skill_descriptions:
        for complexity in complexity_levels:
            for domain in domains:
                try:
                    instructions = generate_skill_specific_instructions(skill, complexity, domain)
                    log_output('Original Seed Instructions', instructions)
                    generated_content = llm_infer(instructions)
                    log_output('Original Generated Content', generated_content)

                    steps = [generated_content]
                    # Self-refinement loop
                    for i in range(2):
                        generated_content = refine_problem(generated_content, i+1)
                        steps.append(generated_content)

                    final_synth_ex = json.loads(generated_content)
                    next_out = send_eval_query(json.dumps(final_synth_ex['input_documents']), final_synth_ex['instruction'])

                    #next_out['ground_truth'] = final_synth_ex['ground_truth']
                    #steps.append(json.dumps(next_out))
                    # Generate HTML
                    html_content = generate_html(GLOBAL_OUTPUT)

                    # Hash the data to create a filename

                    from pathlib import Path
                    
                    Path(config.exp_dir).mkdir(parents=True, exist_ok=True)

                    filename = f"{config.exp_dir}/{skill[0:8]}_{complexity[0:8]}_{domain[0:8]}_{hash_data(GLOBAL_OUTPUT)}.html".replace(" ", "")

                    # Write HTML to file
                    with open(filename, 'w') as file:
                        file.write(html_content)
                    print(f"HTML file created: {filename}")
                    GLOBAL_OUTPUT.clear()
                    counter += 1
                except Exception as e:
                    print('ERRORRRRR\n\n')
                    print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from config import ParseKwargs, Config

    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})

    args = parser.parse_args()
    import json
    print(args)
    config = Config(args.config_files, args.kwargs)

    generated_examples = generate_examples(config)
    import pdb
    pdb.set_trace()