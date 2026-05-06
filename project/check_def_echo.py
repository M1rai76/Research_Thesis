import json

path = "samples/llama-33-70b-versatile_t02_detailed_spec.jsonl"

count = 0
examples = []

with open(path, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        completion = row["completion"].lstrip()
        if completion.startswith("def "):
            count += 1
            if len(examples) < 10:
                examples.append((row["task_id"], completion[:120]))

print(f"Starts with def: {count}")

for task_id, text in examples:
    print(f"\n--- {task_id} ---")
    print(repr(text))