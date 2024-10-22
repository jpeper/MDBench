import json
from statistics import mean, stdev
from utils import DATA_PATH, METADATA_PATH

document_scores = []
table_scores = []

score_dicts = []

for path in (DATA_PATH / "markdown_567_full_out").iterdir():
    with open(path / "full_log.json") as f:
        full_log = json.load(f)

        document_scores.append(int(full_log["DOCUMENT Reasoning Results -- GPTEval Scalar Score"]))
        table_scores.append(int(full_log["TABLE Reasoning Results -- GPTEval Scalar Score"]))

        score_dicts.append({
            "dir_name": path.parts[-1],
            "document_score": document_scores[-1],
            "table_score": table_scores[-1]
        })

stats_dict = {
    "document_scores": {
        "mean": mean(document_scores),
        "min": min(document_scores),
        "max": max(document_scores),
        "stdev": stdev(document_scores)
    },
    "table_scores": {
        "mean": mean(table_scores),
        "min": min(table_scores),
        "max": max(table_scores),
        "stdev": stdev(table_scores)
    }
}

with open((METADATA_PATH / "stats.json"), "w") as f:
    json.dump(stats_dict, f, indent=4) 

with open((METADATA_PATH / "scores.json"), "w") as f:
    json.dump(score_dicts, f, indent=4)