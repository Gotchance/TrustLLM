import os
import json
from trustllm.task.pipeline import run_safety


model_name="deepseek-r1-1.5b"

# 保存先ディレクトリ
save_dir = f"evaluation_results/{model_name}"
os.makedirs(save_dir, exist_ok=True)

safety_results = run_safety(
    jailbreak_path=f"generation_results/{model_name}/safety/jailbreak.json",
    exaggerated_safety_path=f"generation_results/{model_name}/safety/exaggerated_safety.json",
    misuse_path=f"generation_results/{model_name}/safety/misuse.json",
    toxicity_eval=False,
    # toxicity_path=f"generation_results/{model_name}/safety/jailbreak.json",
    jailbreak_eval_type="single"    # total or single
)

print("=== Easy Pipeline Safety Evaluation ===")
for key, value in safety_results.items():
    print(f"{key}: {value}")

with open(os.path.join(save_dir, "safety_eval.json"), "w") as f:
    json.dump(safety_results, f, indent=2)
