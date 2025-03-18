import subprocess

command = [
    "litgpt", "finetune_lora", "Qwen/Qwen2.5-3B-Instruct",
    "--precision", "bf16-true",
    "--data", "JSON",
    "--data.json_path", "/teamspace/studios/this_studio/src/finetune_classification.json",
    "--data.val_split_fraction", "0.1",
    "--out_dir", "out/custom-model"
]


result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

print("Standard Output:\n", result.stdout)
print("Error Output:\n", result.stderr)
