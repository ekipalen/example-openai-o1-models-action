import os

from dotenv import load_dotenv
from openai import OpenAI
from sema4ai.actions import ActionError, Response, Secret, action

load_dotenv()


@action
def openai_o_1_action(
    prompt: str,
    o1_model: str = "o1-mini",
    openai_api_key: Secret = Secret.model_validate(os.getenv("OPENAI_API_KEY", "")),
) -> Response[dict]:
    """
    Action to get responses from OpenAI o1 models. No conversational history stored so provide it if needed.

    Args:
        prompt (str): Prompt for the o1 LLM.
        o1_model: Defaults to "o1-mini", accepts also "o1-preview" or "gpt-4o".
        openai_api_key: OpenAI API key.

    Returns:
        Response: A Response object containing the result from the LLM.
    """
    if not openai_api_key or not o1_model:
        raise ActionError("OpenAI API key or model missing.")

    if o1_model not in ["o1-mini", "o1-preview", "gpt-4o"]:
        raise ActionError("Invalid model. Must be 'o1-mini', 'o1-preview' or 'gpt-4o'.")

    try:
        client = OpenAI(api_key=openai_api_key.value)

        response = client.chat.completions.create(
            model=o1_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        result = {
            "Response": response.choices[0].message.content,
        }

        return Response(result=result)

    except Exception as e:
        raise ActionError(f"Failed to get response: {str(e)}")
