import re
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from utils import RAW_DATA_PATH

parser = ArgumentParser()
parser.add_argument(
    "-r", "--raw_data_dir", type=str, default="v3_tuesday_50", help="Raw data path"
)
parser.add_argument(
    "-o", "--output_dir", type=str, default="clean", help="Output path"
)
args = parser.parse_args()

def format_headers(text):
    pattern = r'(\*\*.*?\*\*)\n'
    replacement = r'\n\1'
    replaced_text = re.sub(pattern, replacement, text)
    return replaced_text


def get_rid_of_weird_spacing(text):
    # text1 = text.replace(" Current Table:\n ", "Current Table:\n```\n").replace("\n\n Current Question:", "\n```\nCurrent Question:")
    text2 = text.replace("**DOCUMENT Reasoning Results -- GPTEval Prediction Evaluation **", "**DOCUMENT Reasoning Results -- GPTEval Prediction Evaluation**")
    return text2


def wrap_json_conditionally(header, next_header, text):
    old_text = file_str.split(f"**{header}**\n")[1].split(f"\n\n**{next_header}**")[0]

    if old_text[0] == "{" or old_text[0] == "[":
        new_text = "```json\n" + old_text + "\n```"
        return text.replace(old_text, new_text)

    return text


def create_json_blocks(text):
    text1 = wrap_json_conditionally("Generated Edits Rationale", "Resultant Generated Table Markdown", text) 
    text2 = wrap_json_conditionally("Example Difficulty", "Validity Scalar", text1)
    text3 = wrap_json_conditionally("Document Set Intermediate Calls", "Generated Document Set", text2)
    return text3


def delete_bar_sep_table(text):
    bar_sep_table_start = text.find("**Resultant Generated Table -- Bar-separated**")
    bar_sep_table_end = text.find("**Resultant Generated Question**")
    return text[:bar_sep_table_start] + text[bar_sep_table_end:]


def parse_table(table_str):

    rows = [line for line in table_str.split('\n') if not re.match(r'^[\+\-\|\= ]+$', line)]

    print(rows)
    
    header = rows[0].split('|')[1:-1]
    header = [item.strip() for item in header]

    data = []
    for row in rows[1:]:
        columns = row.split('|')[1:-1]
        columns = [item.strip() for item in columns]
        data.append(columns)
    
    return header, data


def dataframe_to_markdown(df):
    return df.to_markdown(index=False)


def create_markdown_table(text, split_begin, split_end):
    markdown_table = text.split(f"**{split_begin}**\n")[1].split(f"\n\n**{split_end}**")[0]
    # header, data = parse_table(markdown_table)
    # df = pd.DataFrame(data, columns=header)
    # new_markdown_table = dataframe_to_markdown(df)
    new_markdown_table = markdown_table.replace(":", "")
    return text.replace(markdown_table, f"\n{new_markdown_table}")

raw_data_path = RAW_DATA_PATH / args.raw_data_dir

for path in raw_data_path.iterdir():

    try:
        with open(path / "full_log.txt") as f:
            file_str = f.read()
    except:
        continue

    file_str = format_headers(file_str)
    file_str = get_rid_of_weird_spacing(file_str)
    file_str = create_json_blocks(file_str)
    file_str = delete_bar_sep_table(file_str)
    file_str = create_markdown_table(file_str, "Resultant Generated Table Markdown", "Resultant Generated Question")
    file_str = create_markdown_table(file_str, "Existing Table Info", "Generated Edits Rationale")

    clean_path = Path(RAW_DATA_PATH / Path(args.output_dir) / raw_data_path.name) / f"{path.name}.md"
    clean_path.parent.mkdir(exist_ok=True)

    with open(clean_path, "w") as f:
        f.write(file_str)
