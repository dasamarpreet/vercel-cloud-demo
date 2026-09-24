from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import json
import typing
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 1. Configure API Key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# 2. Define the exact JSON structure (Schema) we want the AI to return
# This is a massive feature to show students: ensuring AI returns predictable data formats.
class WorkshopResponse(typing.TypedDict):
    explanation: str
    key_takeaways: list[str]
    bash_or_python_code_snippet: str
    confidence_score: float

# 3. Initialize the model with a System Instruction
model = genai.GenerativeModel(
    model_name="gemini-flash-lite-latest",
    system_instruction=(
        "You are an expert Senior DevOps and Cloud Computing instructor teaching MCA students. "
        "Keep your explanations highly technical, accurate, and concise. "
        "Always provide practical code or CLI command examples."
    )
)

@app.route('/')
def home():
    return jsonify({"message": "Advanced GenAI Workshop API is Live!"})

@app.route('/api/chat', methods=['GET'])
def chat():
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "Provide a prompt query parameter"}), 400
    
    try:
        # 4. Use GenerationConfig to control AI behavior
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=WorkshopResponse,
                temperature=0.3,          # Lower temperature (0.0 - 0.3) for factual, deterministic coding answers
                top_p=0.8,                # Nucleus sampling - limits token choices to top 80% probable tokens
                top_k=40,                 # Limits token choices to top 40 absolute tokens
                max_output_tokens=800     # Prevents the API from running indefinitely and burning tokens
            )
        )
        
        # Because we used response_mime_type="application/json", response.text is a pure JSON string.
        # We parse it into a Python dictionary before returning it via Flask so it formats beautifully.
        ai_data = json.loads(response.text)
        
        return jsonify({
            "status": "success",
            "metadata": {
                "model_used": "gemini-flash-lite-latest",
                "temperature_applied": 0.3
            },
            "data": ai_data
        })
        
    except Exception as e:
        # Catching and returning the error helps students debug cloud issues
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
