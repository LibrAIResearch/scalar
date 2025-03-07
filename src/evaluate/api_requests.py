from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3), reraise=False)
def chat_completion_request(client, messages, tools=None, tool_choice=None, key_pos=None, model="gpt-3.5-turbo", stop=None, **args):
    use_messages = []
    for message in messages:
        if not ("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)
    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        json_data.update({"stop": stop})
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})

    try:
        # print(json_data)
        if "response_format" in args:
            openai_response = client.beta.chat.completions.parse(**json_data)
        else:
            openai_response = client.chat.completions.create(**json_data)
        json_data = openai_response.dict()
        return json_data
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        raise e
    