import requests
import json
import sseclient

API_URL = "https://custom-llm-8dod.onrender.com/chat/completions"

def send_chat_request(message, model="llama-3.3-70b-versatile"):
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'system', 'content': 'you must check availability when asked\n\nKnowledge Base Results:\nAn unexpected error occurred during search: Event loop is closed\n\n'}, {'role': 'assistant', 'content': 'Hello.'}, {'role': 'user', 'content': 'Hello?'}, {'role': 'assistant', 'content': "I'm here to help. Is there something I can help you with? Or would you like to check the availability for an appointment?"}, {'role': 'user', 'content': 'I want you to kiss me.'}], 'temperature': 0.5, 'tools': [{'type': 'function', 'function': {'name': 'check_availability', 'description': 'use this to check the availability for appointments', 'parameters': {'type': 'object', 'required': ['end_time', 'start_time', 'appointment_type'], 'properties': {'end_time': {'type': 'string', 'description': "the end time range of when you want to look. 'Thursday 11pm', 'Next Tuesday 3pm', 'June 23rd 3pm'. If you want to check the whole day just simply make the time part of the date 11pm."}, 'start_time': {'type': 'string', 'description': "the start time of when you want to look for availability. i.e. 'Thursday 8am',  'Next Tuesday 11am', 'June 25th 8am'. If you want to check the availability for the whole day make the time portion of start_time '8am'"}, 'appointment_type': {'enum': ['consultation_andrew', 'consultation_lucy', 'consultation_andreea'], 'type': 'string', 'description': 'the type of appointment being booked'}}}}}], 'max_tokens': 250, 'stream': True}
    
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
