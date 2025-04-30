import requests
import json
import sseclient

API_URL = "https://custom-5ahbkbh3u-hugopod-artiloaicoms-projects.vercel.app/chat/completions"

def send_chat_request(message, model="llama-3.3-70b-versatile"):
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": message}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    response = requests.post(API_URL, headers=headers, json=data, stream=True)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    # Parse the SSE response
    client = sseclient.SSEClient(response)
    full_response = ""
    
    for event in client.events():
        if event.data == "[DONE]":
            break
        
        try:
            chunk_data = json.loads(event.data)
            content = chunk_data.get("content", "")
            full_response += content
            print(content, end="", flush=True)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {event.data}")
    
    print("\n\nFull response:", full_response)
    return full_response

if __name__ == "__main__":
    user_message = input("Enter your message: ")
    send_chat_request(user_message)
