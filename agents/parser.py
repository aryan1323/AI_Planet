import json
import re
from langchain_core.prompts import PromptTemplate
from .base import get_llm

PARSER_PROMPT = PromptTemplate(
    input_variables=["input_text"],
    template="""
    You are a Math Parser. 
    1. Fix any typos in the text below.
    2. Format it as a clean math problem.
    3. Output STRICT JSON.
    
    Input: "{input_text}"
    
    JSON Schema:
    {{
        "problem_text": "The full cleaned text of the problem",
        "topic": "Guess the topic (Algebra, Probability, Calculus, etc)",
        "needs_clarification": false
    }}
    
    Do NOT add markdown. Output ONLY the JSON object.
    """,
)


def run_parser_agent(raw_text):
    text = raw_text.strip()
    if len(text) > 100 or "balls" in text or "probability" in text:
        return {
            "problem_text": text,
            "topic": "Probability/General",
            "needs_clarification": False,
        }
    llm = get_llm()
    try:
        response = llm.invoke(PARSER_PROMPT.format(input_text=text))

        content = response if isinstance(response, str) else response.content

        clean_json = content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(clean_json)

        if "problem_text" not in parsed:
            parsed["problem_text"] = text

        return parsed

    except Exception as e:
        print(f"Parsing failed, using raw input. Error: {e}")
        return {"problem_text": text, "topic": "General", "needs_clarification": False}
