import openai

openai.api_key = 'INPUT_API_YOUR_KEY'

# Define your conversation
def proceed_explanation(data):
    conversation = [
        {'role': 'system', 'content': 'You are an expert in CLUB MANAGEMENT. Why this customer is churned?'},
        {'role': 'user', 'content': data}
        ]
    # Send the message to the ChatGPT API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Extract the assistant's reply
    summary = response['choices'][0]['message']['content']
    return summary
