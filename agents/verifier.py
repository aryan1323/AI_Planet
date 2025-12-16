from .base import get_llm
from langchain_core.prompts import PromptTemplate

VERIFIER_PROMPT = PromptTemplate(
    input_variables=["problem_text", "solution"],
    template="""
    Verify this math solution.
    Problem: {problem_text}
    Solution: {solution}
    Strictly check for logical errors. Output 'VERIFIED_CORRECT' or 'VERIFIED_INCORRECT'.
    """
)

def run_verifier_agent(problem_text, solution):
    llm = get_llm()
    response = llm.invoke(VERIFIER_PROMPT.format(problem_text=problem_text, solution=solution))
    return response if isinstance(response, str) else response.content