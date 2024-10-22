import os
import asyncio
import time

from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Replicate


from langchain.schema import HumanMessage, SystemMessage
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

from tqdm import tqdm

from prompts import CoT
from utils import CONTEXT_KEY, INPUT_KEY, OUTPUT_KEY, API_INSTRUCTION_IDX

def is_concurrent(name):
    return "gemini" not in name.lower() and "multiagent" not in name.lower() and "claude123" not in name.lower()

def get_formatted_examples(examples):
    formatted_examples = []
    for ex in examples:
        formatted_example = []
        formatted_example.extend(
            [
                HumanMessage(content=f"{CONTEXT_KEY} {ex['context']}"),
                HumanMessage(content=f"{INPUT_KEY} {ex['question']}"),
                HumanMessage(content=f"{OUTPUT_KEY} {ex['ground_truth']}"),
            ]
        )

        formatted_examples.extend(formatted_example)

    return formatted_examples


def get_api_prompts(prompts, prompt_instr, x_shot_prompt):
    api_prompts = []

    for prompt in prompts:
        api_prompt = [prompt_instr]
        examples = get_formatted_examples(prompt["examples"]) if "zero" not in x_shot_prompt else []
        api_prompt.extend(examples)

        if "CoT" in x_shot_prompt:
            api_prompt.extend(
                [
                    HumanMessage(content=f"{CONTEXT_KEY} {prompt['context']}"),
                    HumanMessage(content=f"{INPUT_KEY} {prompt['question']}"),
                    HumanMessage(content=CoT),
                    HumanMessage(content=OUTPUT_KEY),
                ]
            )
        else:
            api_prompt.extend(
                [
                    HumanMessage(content=f"{CONTEXT_KEY} {prompt['context']}"),
                    HumanMessage(content=f"{INPUT_KEY} {prompt['question']}"),
                    HumanMessage(content=OUTPUT_KEY),
                ]
            ) 

        api_prompts.append(api_prompt)

    return api_prompts


def get_llm(model, max_tokens):
    # model_kwargs = {}
    # if use_json:
    #     model_kwargs = {"response_format": {"type": "json_object"}}
    temperature = 0

    if "gpt" in model:
        llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=7,
            # model_kwargs=model_kwargs
        )
    elif "claude" in model:
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=7,
        )
    elif "gemini" in model:
        from warnings import filterwarnings
        # Suppress the specific warning
        filterwarnings("ignore", category=UserWarning, message="Convert_system_message_to_human will be deprecated!")

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=7,
            convert_system_message_to_human=True,
        )
    elif "llama" in model:
        llm = Replicate(model=model, model_kwargs={"temperature": 0.0, "max_length": max_tokens, "top_p": 1})
    elif "multiagent" in model:
        return ""
    else:
        raise Exception("Not a valid model.")
    return llm



def query_api(
    prompts,
    model='gpt-4o',
    name='gpt-4o',
    max_tokens=1024,
    serial=False,
    parallel=True,
    x_shot_prompt="zero-shot",
    num_prompts=None,
    # use_json=False,
):

    llm = get_llm(model, max_tokens)

    if not num_prompts:
        num_prompts = len(prompts) - 1

    print(f"{num_prompts} examples.")

    prompt_instr = SystemMessage(content=prompts[API_INSTRUCTION_IDX])
    prompts = prompts[1 : num_prompts + 1]

    api_prompts = get_api_prompts(prompts, prompt_instr, x_shot_prompt)

    if "multiagent" in model:
        flattened_prompts = ["\n".join([elt.content for elt in ex]) for ex in api_prompts]
        from multi_agent.run_multi_agent import pythonic_chatloop

        agent_map = {
            "multiagent_auto": "<todo add generic path>",
            "multiagent_manual": "<todo add generic path>"
        }
        multi_agent_config = agent_map[model]

        results = [pythonic_chatloop(multi_agent_config, example) for example in flattened_prompts]
        return results

    def invoke_serially(llm, prompts):

        # pdb.set_trace()
        # print('here')
        resp = [
            llm.invoke(prompt)
            for prompt in tqdm(
                prompts,
                desc=f"Serially invoking {name}",
                unit="prompt",
            )
        ]
        if "llama" not in name.lower():
            filtered = [rev.content for rev in resp]
            return filtered
        return resp

    async def invoke_concurrently_batch(llm, prompts):
        results = []
        batch_size = 10  # Adjust as needed

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            try:
                # Process the entire batch
                resp = await llm.abatch(
                    [prompt for prompt in tqdm(batch, desc=f"Concurrently invoking {name}", unit="prompt")],
                    {"max_concurrency": 5}
                )
                # For non-Llama models, filter responses
                if "llama" not in name.lower():
                    results.extend([r.content for r in resp])
                else:
                    results.extend(resp)
            except Exception as e:
                if 'content filter being triggered' in str(e) or 'was filtered due to' in str(e):
                    print(f"Batch starting at index {i} triggered content filter. Retrying individually.")
                # Retry each prompt individually if batch fails
                for idx, prompt in enumerate(batch):
                    try:
                        resp = await llm.abatch(
                            [prompt],
                            {"max_concurrency": 1}
                        )
                        if "llama" not in name.lower():
                            results.append(resp[0].content)
                        else:
                            results.append(resp[0])
                    except Exception as e:
                        if 'content filter being triggered' in str(e) or 'was filtered due to' in str(e):
                            print(f"Prompt at index {i+idx} triggered content filter.")
                            # Return "No Response" for content-filtered prompts
                            results.append("No Response")
                        else:
                            print(e)
                            results.append("No Response")

        return results

    if serial:
        s = time.perf_counter()
        output = invoke_serially(llm, api_prompts)
        elapsed = time.perf_counter() - s
        print(
            "\033[1m"
            + f"Serially executed {num_prompts} examples in {elapsed:0.2f} seconds."
            + "\033[0m"
        )

    if parallel:
        s = time.perf_counter()
        output = asyncio.run(invoke_concurrently_batch(llm, api_prompts))
        elapsed = time.perf_counter() - s
        print(
            "\033[1m"
            + f"Concurrently executed {num_prompts} examples in {elapsed:0.2f} seconds."
            + "\033[0m"
        )

    return output
