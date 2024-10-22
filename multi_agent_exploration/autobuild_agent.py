import os
from dotenv import load_dotenv
from models import gpt4o_config
import json

from autogen.agentchat.contrib.agent_builder import AgentBuilder

import sys
builder = AgentBuilder(config_file_or_env='builder_config.json', builder_model='gpt-4o', agent_model='gpt-4o')

building_task = json.load(open(sys.argv[1]))["prompt"]

agent_list, agent_configs = builder.build(building_task, gpt4o_config, coding=True)

import autogen

def start_task(execution_task: str, agent_list: list, llm_config: dict):
    config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-4-1106-preview"]})

    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}
    )
    agent_list[0].initiate_chat(manager, message=execution_task)

saved_path = builder.save(sys.argv[2])


