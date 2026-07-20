from sympy import sympify, solve

def math_tool(expression):
    try:
        expr = sympify(expression)
        result = expr.evalf()
        return {"result": float(result)}
    except Exception as e:
        return {"error": f"Invalid expression: {str(e)}"}
    

# Example use of agent
# print(math_tool("6.626e-34 / (2 * 3.14159)"))