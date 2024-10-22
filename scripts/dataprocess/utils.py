from pathlib import Path
from sys import path

path.insert(1, "./scripts/")
from llm.utils import construct_dataset

RAW_DATA_PATH = Path("data/raw_data")
METADATA_PATH = RAW_DATA_PATH / Path("metadata")

TABLE_VALIDITY = "Valid Table [0: no, 1: yes]"
DOCUMENT_VALIDITY = "Valid Docs [0: no, 1: yes]"
EXAMPLES = "Example ID"