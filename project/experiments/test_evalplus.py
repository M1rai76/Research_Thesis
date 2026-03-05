from evalplus.data import get_human_eval_plus

problems = get_human_eval_plus()

print("Number of tasks:", len(problems))

first_key = list(problems.keys())[0]
print("First task:", first_key)

print("Prompt preview:")
print(problems[first_key]["prompt"][:200])

task = problems[first_key]
print(task.keys())