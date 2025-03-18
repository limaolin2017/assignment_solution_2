from litgpt import LLM

llm = LLM.load("Qwen/Qwen2.5-3B-Instruct")
text = llm.generate("Fix the spelling: Every fall, the family goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.