from sympy import sympify, solve
# from langchain_classic.tools import tool

# @tool(name_or_callable="calculator")
def math_tool(expression):
    """Useful for performing mathematical calculations."""
    try:
        expr = sympify(expression)
        result = expr.evalf()
        return {"result": float(result)}
    except Exception as e:
        return {"error": f"Invalid expression: {str(e)}"}