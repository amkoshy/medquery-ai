import openai
from openai import AuthenticationError

# openai.api_key is now read from the environment variable OPENAI_API_KEY

try:
    response = openai.models.list()
    print("✅ API Key is valid. Available models:")
    for model in response:
        print(model.id)
except AuthenticationError:
    print("❌ Invalid API key or not authorized.")
except Exception as e:
    print(f"⚠️ Other error: {e}")
