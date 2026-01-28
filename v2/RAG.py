import ollama


def dynamic_rag(user_prompt: str, model: str = "llama3"):
    # 2. Define the 'tool' schema so the model knows what the function does
    tools = [{
        'type': 'function',
        'function': {
            'name': 'retrieve_context',
            'description': 'Get specific internal information to answer questions',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'The search query'},
                },
                'required': ['query'],
            },
        },
    }]

    # 3. Initial Chat Call
    messages = [{'role': 'user', 'content': user_prompt}]
    response = ollama.chat(model=model, messages=messages, tools=tools)

    # 4. Check if the model wants to use the tool
    if response.get('message', {}).get('tool_calls'):
        for tool in response['message']['tool_calls']:
            if tool['function']['name'] == 'retrieve_context':
                # Execute the function
                context = retrieve_context(tool['function']['parameters']['query'])
                
                # Add the tool's result to the conversation
                messages.append(response['message'])
                messages.append({
                    'role': 'tool',
                    'content': context,
                })
        
        # Final call to generate answer based on the retrieved context
        final_response = ollama.chat(model=model, messages=messages)
        return final_response['message']['content']
    
    # If no tool was needed, just return the direct response
    return response['message']['content']

# --- Testing ---
print("Test 1 (General):", dynamic_rag("Hi there!"))
print("\nTest 2 (Specific):", dynamic_rag("What is the secret ingredient?"))