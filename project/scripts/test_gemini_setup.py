import os
from google import genai

def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in PowerShell first."
        )

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with exactly: Gemini setup working"
    )

    print(response.text)

if __name__ == "__main__":
    main()