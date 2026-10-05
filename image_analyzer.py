import requests
import base64


def analyze_image(image_bytes):

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    prompt = """
Look at this image carefully.

First, identify what is shown in the image.

If it is a tourist place or landmark, give your best identification.

Answer in normal sentences. Do NOT use a fixed template.
Do NOT leave blank fields.

Tell me:

1. What is shown in the image?
2. What is the likely name of the place?
3. Which city and country is it in, if you know?
4. What visual details helped you identify it?
5. Why is this place important for tourists?

If you are not sure about the exact place, clearly say:
"I cannot identify the exact place from this image."

Do not invent information.
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [encoded_image],
                "stream": False
            }
        )

        if response.status_code == 200:

            return response.json()["response"]

        else:

            return "LLaVA returned an error: " + response.text

    except Exception as e:

        return "Unable to connect to Ollama/LLaVA: " + str(e)