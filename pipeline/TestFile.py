import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('CHAT_AI_API_KEY')
endpoint = os.getenv('CHAT_AI_ENDPOINT', 'https://llm.cloud.cci.charite.de/v1')
model_name = 'gpt-oss-120b' # Change this to the model you are testing

client = OpenAI(api_key=api_key, base_url=endpoint)

try:
    # Try to list models to see if metadata is exposed
    models = client.models.list()
    print(f"Available models:")
    for m in models.data:
        if model_name in m.id:
            print(f"Found: {m.id}")
            # Some APIs expose context_window in metadata
            if hasattr(m, 'root') and hasattr(m.root, 'context_window'):
                print(f"Context Window: {m.root.context_window}")
            else:
                print("Context window not exposed in model list.")
            
            # Try to get specific model info if supported
            try:
                model_info = client.models.retrieve(m.id)
                print(f"Retrieved info: {model_info}")
            except Exception as e:
                print(f"Could not retrieve specific model info: {e}")
            break
    else:
        print(f"Model '{model_name}' not found in list.")

except Exception as e:
    print(f"Error checking models: {e}")

# Fallback: Try a small generation to see if it accepts high max_tokens
print("\n--- Testing max_tokens limit ---")
try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=20000 # Try a high number
    )
    print(f"Success with 20k tokens. Model likely supports it.")
except Exception as e:
    print(f"Failed with 20k tokens: {e}")
    # Try lower
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=4096
        )
        print(f"Success with 4k tokens. Limit might be 4096.")
    except Exception as e2:
        print(f"Failed even with 4k: {e2}")