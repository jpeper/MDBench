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

from llm_infer import llm_infer

def get_manager(customer: Agent, agent_path, max_round=10):
    """create group chat manager to talk the customer"""

    new_builder = AgentBuilder(config_file_or_env='/Users/jpeper697/Documents/mercury_dumps/MultiAgent/MDDG/multi_agent/builder_config.json')
    agent_list, agent_config = new_builder.load(agent_path)

    import json
    json_config = json.load(open(agent_path))

    new_transitions=None
    transition_type=None
    
    if 'allowed_agent_transitions' in json_config:
        transitions = json_config['allowed_agent_transitions']
        agent_list_json = {elt.name: elt for elt in agent_list}
        
        new_transitions = {}

        for agent, transitions in transitions.items():
            new_transitions[agent_list_json[agent]] = [agent_list_json[transition] for transition in transitions]
        transition_type = "allowed"

    group_chat = autogen.GroupChat(agents=agent_list,
                                   messages=[],
                                   max_round=max_round,
                                   select_speaker_message_template="""You are in a role play game. The following roles are available:
                    {roles}.
                    Read the following conversation.
                    Then select the next role from {agentlist} to play. Only return the role.
                    """,
        select_speaker_prompt_template=(
            "Read the above conversation. Then select the next role from {agentlist} to play. Only return the role."
        ),
        allowed_or_disallowed_speaker_transitions=new_transitions,
        speaker_transitions_type=transition_type
        )

    manager = autogen.GroupChatManager(
        name="group_chat_manager", groupchat=group_chat, llm_config=gpt4o_config
    )

    return manager

def get_customer(human_input_mode="ALWAYS"):
    return UserProxyAgent(
    name="customer",
    system_message=
    """You are a customer.
""",
    llm_config=False,  # no LLM used for human proxy
    code_execution_config=False,  # no code execution for human
    # is_termination_msg=lambda msg: 'Exit_agent' == msg['name'],  # Terminate when the condition is met and user type in empty message
    is_termination_msg=lambda msg: True,  # Always terminate
    human_input_mode=human_input_mode,  # always ask for human input
)

def pythonic_chatloop(agent_path, user_message):
    customer = get_customer("NEVER")
    manager = get_manager(customer, agent_path=agent_path)

    try:
        result = customer.initiate_chat(manager,
                                        message=user_message,
                                        clear_history=False)
        
        output = list(manager._oai_messages.values())
        filtered = [ex["content"] for ex in output[-1] if ex["content"] != "TERMINATE"][-1]
        cleaned = llm_infer("Take this verbose output and concisely return just the final answer -- usually in a couple of sentences" + filtered)
    except Exception as e:
        print(e)
        cleaned = "I don't know how to answer this question"
    
    return cleaned

def manual_chatloop():
    customer = get_customer("NEVER")
    manager = get_manager(customer, agent_path=sys.argv[1])
    # customer.human_input_mode = "NEVER"

    while True:
        import pdb
        pdb.set_trace()
        user_message = "Input: Doc 1: the bananas are red. Doc 2: the strawberries are yellow, Doc 3: the oranges are orange. Question: how many fruits were mentioned?"
        result = customer.initiate_chat(manager,
                                        message=user_message,
                                        clear_history=False)

        print(result.chat_history)
        if 'exit' in result.summary:
            break

if '__main__' == __name__:
    # autogen_chatloop()
    manual_chatloop()


