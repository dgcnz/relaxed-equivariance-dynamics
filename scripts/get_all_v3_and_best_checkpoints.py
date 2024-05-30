import wandb
from collections import defaultdict
from pprint import PrettyPrinter
from itertools import groupby
import yaml

runs = {
    "rgcnn": ["es2ctz7w"],
    "cnn": ["1dq4fnrq"],
}

api = wandb.Api()
entity = "uva-dl2"
project = "wang2024"
printer = PrettyPrinter(indent=4)
runs = api.runs(
    f"uva-dl2/wang2024", filters={"$and": [{"tags": "wang2024"}, {"tags": "final"}]}
)
key_fn = lambda run: (
    run.config["model"]["net"]["_target_"],
    run.config["model"]["net"].get("num_filter_banks", None),
)
runs = sorted(runs, key=key_fn)
g = groupby(runs, key=key_fn)
checkpoints = defaultdict(dict)
for (model_name, num_filter_banks), model_runs in g:
    if num_filter_banks is None:
        num_filter_banks = "null"
    model_runs = sorted(model_runs, key=lambda run: run.summary["test/mae"])
    checkpoints[model_name][f"nfb_{num_filter_banks}"] = []
    for run in model_runs:
        artifacts = run.logged_artifacts()
        test_mae = run.summary["test/mae"]
        best_artifact = next(a for a in artifacts if "best" in a.aliases)
        v3_artifact = next(a for a in artifacts if a.source_version == "v3")
        checkpoints[model_name][f"nfb_{num_filter_banks}"].append(
            {
                "test_mae": test_mae,
                "v3": f"{entity}/{project}/{v3_artifact.name}",
                "best": f"{entity}/{project}/{best_artifact.name}",
            }
        )

with open("checkpoints.yaml", "w") as f:
    f.write(yaml.dump(dict(checkpoints), default_flow_style=False, sort_keys=False))
