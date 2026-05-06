import json

path = "samples/llama-33-70b-versatile_t02_detailed_spec.jsonl"

print("First 5 completions:")
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        print(f"--- {row['task_id']} ---")
        print(repr(row["completion"][:300]))
        print()
        if i >= 4:
            break

print("Empty completion check:")
empty = 0
total = 0

with open(path, encoding="utf-8") as f:
    for line in f:
        total += 1
        row = json.loads(line)
        if not row["completion"].strip():
            empty += 1

print(f"Total: {total}, Empty: {empty}, Valid: {total-empty}")