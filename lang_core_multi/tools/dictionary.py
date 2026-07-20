import httpx
# from langchain_classic.tools import tool

# @tool(name_or_callable="dictionary")
async def dictionary_tool(word):
    """Useful for performing dictionary searches."""
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()[0]
                meaning = data['meanings'][0]

                definition = meaning['definitions'][0]['definition']

                return {
                    "result": f"{word}: {definition}"
                }

            return {"error": "Word not found."}

    except Exception as e:
        return {"error": str(e)}



# Example Usage for your Agent:
# print(dictionary_tool("quantum"))
