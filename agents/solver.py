import math
import re
import sys
import io

import sympy
from sympy import (
    symbols,
    Function,
    dsolve,
    Eq,
    Derivative,
    sin,
    cos,
    tan,
    exp,
    log,
    integrate,
    diff,
    solve,
)

from .base import get_llm
from langchain_core.prompts import PromptTemplate
from rag_engine import retrieve_context

BASIC_CODE_PROMPT = PromptTemplate(
    input_variables=["problem_text"],
    template="""
    You are a Python Math Engineer. Write a script to calculate the numerical answer.
    Problem: {problem_text}
    
    Rules:
    1. Use 'math.comb(n, k)', 'math.factorial(n)', etc.
    2. Print the final result.
    3. Output ONLY code inside ```python ... ```
    """,
)

SYMPY_PROMPT = PromptTemplate(
    input_variables=["problem_text"],
    template="""
    You are a Symbolic Math Expert using the Python library 'sympy'.
    Problem: {problem_text}
    
    Task: Write a Python script using SymPy to solve this analytically.
    
    Reference Guide:
    - Symbols: x, y, z = symbols('x y z')
    - Functions: f = Function('f')(x) or y = Function('y')(x)
    - Derivative: y.diff(x) or diff(f, x)
    - Differential Eq: eqn = Eq(y.diff(x) + y, x)
    - Solve ODE: solution = dsolve(eqn, y)
    - Indefinite Integral: result = integrate(x**2, x)
    - Definite Integral: result = integrate(x**2, (x, 0, 1))
    
    Example for "Solve dy/dx = x + y":
    ```python
    from sympy import symbols, Function, dsolve, Eq
    x = symbols('x')
    y = Function('y')(x)
    eqn = Eq(y.diff(x), x + y)
    sol = dsolve(eqn, y)
    print(sol)
    ```
    
    Rules:
    1. Define all symbols used.
    2. Print the final 'sol' or 'result'.
    3. Output ONLY code inside ```python ... ``` blocks.
    """,
)

SOLVER_PROMPT = PromptTemplate(
    input_variables=["problem_text", "context"],
    template="""
    Solve this math problem step-by-step.
    Context: {context}
    Problem: {problem_text}
    """,
)


def execute_generated_code(code_str, use_sympy=False):
    """
    Executes AI-generated code.
    If use_sympy=True, it injects the entire SymPy library into the execution sandbox.
    """
    if any(x in code_str for x in ["os.", "sys.", "subprocess", "open("]):
        return None, "Unsafe code detected."

    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()

    try:
        safe_globals = {"math": math, "__builtins__": {}}

        if use_sympy:
            safe_globals.update(
                {
                    "sympy": sympy,
                    "symbols": symbols,
                    "Function": Function,
                    "dsolve": dsolve,
                    "Eq": Eq,
                    "Derivative": Derivative,
                    "integrate": integrate,
                    "diff": diff,
                    "solve": solve,
                    "sin": sin,
                    "cos": cos,
                    "tan": tan,
                    "exp": exp,
                    "log": log,
                }
            )

        exec(code_str, safe_globals)
        sys.stdout = old_stdout
        return redirected_output.getvalue().strip(), None
    except Exception as e:
        sys.stdout = old_stdout
        return None, str(e)


def run_solver_agent(problem_text, topic):
    llm = get_llm()

    if topic == "CALCULUS" or any(
        w in problem_text.lower()
        for w in ["differential", "derivative", "integrate", "integral", "dy/dx"]
    ):
        code_response = llm.invoke(SYMPY_PROMPT.format(problem_text=problem_text))
        content = (
            code_response if isinstance(code_response, str) else code_response.content
        )
        match = re.search(r"```python(.*?)```", content, re.DOTALL)

        if match:
            code = match.group(1).strip()
            result, error = execute_generated_code(code, use_sympy=True)
            if result:
                return (
                    f"**Symbolic Solution (via SymPy):**\n`{result}`\n\n**Code:**\n```python\n{code}\n```",
                    ["Generated SymPy Code"],
                )
            else:
                pass

    elif topic in ["PROBABILITY", "ALGEBRA", "LINEAR_ALGEBRA"]:
        code_response = llm.invoke(BASIC_CODE_PROMPT.format(problem_text=problem_text))
        content = (
            code_response if isinstance(code_response, str) else code_response.content
        )
        match = re.search(r"```python(.*?)```", content, re.DOTALL)

        if match:
            code = match.group(1).strip()
            result, error = execute_generated_code(code, use_sympy=False)
            if result:
                return (
                    f"**Calculated Answer:**\n`{result}`\n\n**Code:**\n```python\n{code}\n```",
                    ["Generated Python Code"],
                )

    context = retrieve_context(problem_text)
    context_str = "\n".join(context)
    response = llm.invoke(
        SOLVER_PROMPT.format(problem_text=problem_text, context=context_str)
    )

    return (response if isinstance(response, str) else response.content), context
