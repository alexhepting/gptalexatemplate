import openai
import os
import spacy
from flask_ask import Ask, statement

openai.api_key = os.environ["OPENAI_API_KEY"]

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

def generate_response(prompt, previous_response):
    model_engine = "text-davinci-002"
    prompt = (f"{prompt}\n{previous_response}")
    stop = previous_response

    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=stop,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message

def extract_information(text):
    # Use the spacy model to extract entities from the text
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def classify_intent(text):
    # Use the spacy model to classify the intent of the text
    doc = nlp(text)
    intent = doc.cats
    return intent

def create_response(prompt, previous_response):
    response, entities, intent = handle_intent(prompt, previous_response)
    return statement(response).simple_card(title='GPT-3 Response', content=response)

app = Flask(__name__)
ask = Ask(app, '/')

@ask.intent('ConversationIntent')
def conversation(prompt):
    return create_response(prompt, previous_response)

if __name__ == '__main__':
    app.run()
