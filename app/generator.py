from openai import OpenAI

from .config import GROQ_API_KEY, LLM_MODEL


class Generator:

    def __init__(self):

        if not GROQ_API_KEY:

            raise ValueError(
                "GROQ_API_KEY not found."
            )

        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

    def generate(self, question, matched_items):

        system_message = """
You are a helpful assistant for question answering.

The context provided below was retrieved from a PDF document.

Answer the user's question using only the provided context.

Do not invent information.

If the answer cannot be found in the provided context,
say that the information was not found in the document.
"""

        context_parts = []

        for item in matched_items:

            if item["type"] in [
                "text",
                "table"
            ]:

                context_parts.append(
                    f"[Page {item['page']}]\n"
                    f"{item['text']}"
                )

        context = "\n\n".join(
            context_parts
        )

        user_message = f"""
Retrieved document context:

{context}

Question:
{question}
"""

        response = self.client.responses.create(
            model=LLM_MODEL,
            input=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        return response.output_text