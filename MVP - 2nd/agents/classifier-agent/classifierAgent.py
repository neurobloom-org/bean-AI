import ollama

def classify_intent(user_input):
    # This calls the model you uploaded to Hugging Face
    model_name = "hf.co/pieterszharsh/llama-3.2-1b-intent-classifier"
    
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': user_input,
        },
    ])
    
    # Return the clean intent label (e.g., "THERAPEUTIC")
    return response['message']['content'].strip()

# Quick test logic
if __name__ == "__main__":
    test_query = "I'm feeling very stressed about my exams."
    intent = classify_intent(test_query)
    print(f"User Query: {test_query}")
    print(f"Detected Intent: {intent}")