import ollama

def classify_intent(user_input):
    model_name = "bean-classifier" 
    
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': user_input,
        },
    ])
    
    return response['message']['content'].strip()

# Interactive Logic
if __name__ == "__main__":
    print("--- Bean AI Intent Classifier ---")
    print("Type 'exit' to quit the program.")
    
    while True:
        # This line waits for YOU to type something in the terminal
        user_text = input("\nEnter your message: ")
        
        if user_text.lower() == 'exit':
            break
            
        intent = classify_intent(user_text)
        
        print(f"Detected Intent: {intent}")