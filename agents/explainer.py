from .base import get_llm
from langchain_core.prompts import PromptTemplate

EXPLAINER_PROMPT = PromptTemplate(
    input_variables=["problem_text", "solution"],
    template="""
    Explain this solution to a student.
    Problem: {problem_text}
    Solution: {solution}
    """
)

def run_explainer_agent(problem_text, solution):
    llm = get_llm()
    response = llm.invoke(EXPLAINER_PROMPT.format(problem_text=problem_text, solution=solution))
    return response if isinstance(response, str) else response.content