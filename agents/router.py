from .base import get_llm
from langchain_core.prompts import PromptTemplate

ROUTER_PROMPT = PromptTemplate(
    input_variables=["problem_text"],
    template="""
    You are a Senior Math Classifier.
    Classify the following math problem into ONE of these categories:
    [ALGEBRA, CALCULUS, PROBABILITY, LINEAR_ALGEBRA, GEOMETRY, STATISTICS, NUMBER_THEORY]
    
    Problem: "{problem_text}"
    
    Rules:
    - If it involves derivatives, integrals, limits, or rates of change -> CALCULUS
    - If it involves matrices, vectors, eigenvalues -> LINEAR_ALGEBRA
    - If it involves chance, dice, coins, distributions -> PROBABILITY
    - If it involves shapes, angles, areas, volumes -> GEOMETRY
    - If it involves equations, polynomials, complex numbers -> ALGEBRA
    
    Output ONLY the category name. Do not explain.
    """,
)


def run_router_agent(problem_text):
    text_lower = problem_text.lower()

    if any(
        x in text_lower
        for x in [
            "dy/dx",
            "integrate",
            "derivative",
            "area under curve",
            "l'hopital",
            "taylor series",
        ]
    ):
        return "CALCULUS"

    if any(
        x in text_lower
        for x in [
            "eigenvalue",
            "eigenvector",
            "determinant",
            "matrix multiplication",
            "row echelon",
            "linear map",
        ]
    ):
        return "LINEAR_ALGEBRA"

    if any(
        x in text_lower
        for x in [
            "probability",
            "conditional distribution",
            "random variable",
            "bayes",
            "variance",
            "standard deviation",
        ]
    ):
        return "PROBABILITY"

    if any(
        x in text_lower
        for x in [
            "triangle",
            "circle",
            "radius",
            "hypotenuse",
            "volume of",
            "surface area",
            "perimeter",
        ]
    ):
        return "GEOMETRY"

    try:
        llm = get_llm()
        response = llm.invoke(ROUTER_PROMPT.format(problem_text=problem_text))

        category = response if isinstance(response, str) else response.content
        category = category.strip().upper().replace(".", "")

        valid_topics = [
            "ALGEBRA",
            "CALCULUS",
            "PROBABILITY",
            "LINEAR_ALGEBRA",
            "GEOMETRY",
            "STATISTICS",
            "NUMBER_THEORY",
        ]

        if any(topic in category for topic in valid_topics):
            for topic in valid_topics:
                if topic in category:
                    return topic

        return "ALGEBRA"

    except Exception as e:
        print(f"Router LLM Error: {e}")
        return "ALGEBRA"
