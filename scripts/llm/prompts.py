instr = "You will be presented with a question and a context.\nYour should answer the question based on the context. The last thing you generate should be \"ANSWER: (your answer here)"
# CoT = "Let's think step by step. Output in the following format: <rationale>####<answer>"
# CoT = """Output steps:
# 1. Carefully read the question and context.
# 2. Give a step by step rationale to find the answer to the question based on the given context.
# 3. Output the answer to the question.
# """
CoT = "Explain your reasoning step by step before you answer. Finally, the last thing you generate should be \"ANSWER: (your answer here)\""

PROMPTS = {
    "zero-shot": instr,
    "zero-shot-CoT": CoT,
    "few-shot": f"{instr}\nHere are some examples:",
    "few-shot-CoT": f"{CoT}\nHere are some examples:",
}
