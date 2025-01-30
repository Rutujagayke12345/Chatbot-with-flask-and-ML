from scripts.intent_recognition import predict_intent

def get_chatbot_response(user_input):
    intent = predict_intent(user_input)
    # Here you could define more logic for selecting responses
    return f"Predicted intent: {intent}"
