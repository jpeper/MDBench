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
from utils import ENV_PATH
from dotenv import load_dotenv

import asyncio
import logging
from langchain.schema import HumanMessage

# Configure logging
def llm_infer(ex, model='gpt4o'):
    """
    Input: the llm prompt (string)
    Output: llm response (string?)
    Used for general-purpose gpt-4o LLM eval/generation inference
    """
    llm = AzureChatOpenAI(
        openai_api_version="2024-02-15-preview",
        azure_deployment='https://jpeper-mdreasoning.openai.azure.com',
        temperature=0.0,
        max_tokens = 4096,
        max_retries=7
    )

    async def invoke_concurrently_batch(llm, exs):
        batch_size = 10  # Adjust as needed
        results = []
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i+batch_size]
            try:
                # Attempt to process the entire batch
                resp = await llm.abatch(
                    [[HumanMessage(content=query)] for query in batch],
                    {'max_concurrency': 10}
                )
                # Append the responses to the results
                results.extend([r.content for r in resp])
            except Exception as e:
                if 'content filter being triggered' in str(e) or 'was filtered due to' in str(e):
                    print(f"Batch starting at index {i} triggered content filter. Retrying individually.")
                # If batch fails, process each query individually
                for idx, query in enumerate(batch):
                    try:
                        resp = await llm.abatch(
                            [[HumanMessage(content=query)]],
                            {'max_concurrency': 1}
                        )
                        results.append(resp[0].content)
                    except Exception as e:
                        if 'content filter being triggered' in str(e) or 'was filtered due to' in str(e):
                            print(f"Query at index {i+idx} triggered content filter.")
                            # Return "No Response" for content-filtered queries
                            results.append("No Response")
                        else:
                            print(e)
                            results.append("No Response")
    
        return results

    s = time.perf_counter()

    num_examples = 1
    if isinstance(ex, list):
        output = asyncio.run(invoke_concurrently_batch(llm, ex))
        num_examples = len(ex)
    else:
        output = asyncio.run(invoke_concurrently_batch(llm, [ex]))
    elapsed = time.perf_counter() - s
    print("\033[1m" + f"Concurrently executed {num_examples} examples in {elapsed:0.2f} seconds." + "\033[0m")

    if isinstance(ex, list):
        return output
    else:
        return output[0]