from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

app = Flask(__name__)

# Feingetuntes Modell laden
model_path = "./fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    notes = data.get("notes", "")
    response = generator(notes, max_length=100, num_return_sequences=1)
    generated_text = response[0]["generated_text"]
    return jsonify({"text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
