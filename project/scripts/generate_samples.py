from evalplus.data import get_human_eval_plus
import json

problems = get_human_eval_plus()

samples = []

for task_id in problems.keys():
    samples.append({
        "task_id": task_id,
        "completion": "def solution(): pass"
    })

with open("samples/samples.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print("Samples file created.")