import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs(
    f"uva-dl2/wang2024", filters={"$and": [{"tags": "wang2024"}, {"tags": "sharpness"}]}
)

run_data = [
    {
        "model_name": run.config["model_name"].split(".")[-1],
        "num_filter_banks": run.config["num_filter_banks"],
        "artifact_version": run.config["artifact_version"],
        "checkpoint": "Early" if run.config["artifact_version"] == "v3" else "Best",
        "sharpness": run.summary.get("sharpness", None),
    }
    for run in runs
]
df = pd.DataFrame(run_data)
print(df)
fig, ax = plt.subplots()

pink = "#E87B9F"
cyan = "#229487"
grey = "#A1A9AD"

model_name_to_color = {
    "GCNNOhT3": cyan,
    "RGCNNOhT3": pink,
    "SuperResCNN": grey,
}
checkpoint_order = ["Early", "Best"]
grouped = df.groupby(["checkpoint", "model_name"])
mean_sharpness = grouped.sharpness.mean().unstack()
std_sharpness = grouped.sharpness.std().unstack()

mean_sharpness = mean_sharpness.reindex(checkpoint_order)
std_sharpness = std_sharpness.reindex(checkpoint_order)
colors = [model_name_to_color[model_name] for model_name in mean_sharpness.columns]
ax.set_ylabel("Sharpness")
mean_sharpness.plot(kind="bar", yerr=std_sharpness, color=colors, alpha=0.9, ax=ax)
plt.savefig("sharpness.png", dpi=300)
plt.show()
