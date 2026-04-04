from flask import Flask, request, jsonify, render_template
from rag_pipeline import create_vector_db, load_llm

app = Flask(__name__)

#  Load once
print(" Loading Vector DB...")
vector_db = create_vector_db()

print(" Loading LLM...")
llm = load_llm()

print(" App is ready!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_question = data.get("message")

        if not user_question:
            return jsonify({"reply": "Please enter a valid question."})

        print("User:", user_question)

        #  FIXED INDENTATION
        docs = vector_db.max_marginal_relevance_search(user_question, k=5)

        if not docs:
            return jsonify({"reply": "No relevant info found in PDF."})

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are a helpful AI assistant.

        Answer ONLY using the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        response = llm.invoke(prompt)

        #  Safe response extraction
        if isinstance(response, dict) and "generated_text" in response:
            answer = response["generated_text"]
        elif hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

        print("Bot:", answer)

        return jsonify({"reply": answer})

    except Exception as e:
        print(" Error:", str(e))
        return jsonify({"reply": "Something went wrong."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)