import gradio as gr
import pickle
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load OpenAI credentials
with open("D:\\Facultate\\Venice Data Week\\openaikey.txt", 'r') as f:
    cont = f.readlines()
params = dict((v.replace('\n', '').split('=')) for v in cont)

# Load mapped entities
with open("mapped_names.pkl", "rb") as f:
    loaded_entities = pickle.load(f)
    loaded_entities = {k: v for d in loaded_entities for k, v in d.items()}

# System prompt
base_system_msg = (
    "Given a place and some entities, build a short explanatory text description of the place, using "
    "the entities. The place is preceded by the keyword 'Place' and the entities are listed "
    "after the keyword 'Entities' and separated by the special character #."
)


# Model
def get_client_with_different_temp(temp: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        openai_organization=params['organization_id'],
        api_key=params['api_key'],
        model='gpt-4o',
        temperature=temp
    )


# Wrapped function
def ask_about_place(place):
    if place not in loaded_entities:
        return "🛑 Sorry, no historical data found for this place."

    most_connected_entities = "#".join(loaded_entities[place]).strip("#")
    in_text = f"Place:{place}\n Entities:{most_connected_entities}"

    guess_prompt = ChatPromptTemplate.from_messages([
        ("system", base_system_msg),
        ("human", "{Text Input}"),
    ])

    chatgpt_llm = get_client_with_different_temp()
    guess_agent = guess_prompt | chatgpt_llm

    try:
        response = guess_agent.invoke({'Text Input': in_text})
        return response.content
    except Exception as e:
        return f"⚠️ An error occurred: {e}"


# Chatbot interface layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="background-color:#f9f1e7;padding:20px;border-radius:10px;text-align:center;">
            <h1 style="font-family:serif;color:#5a3e36;">🕰️ Venice Place Explainer</h1>
            <p style="font-family:Georgia;font-size:16px;color:#5a3e36;">
                Enter a historical Venetian place to uncover a brief explanation based on notable entities.
                This chatbot draws upon graph-based reasoning and language models to bring the past to life.
            </p>
        </div>
        """)

    chatbot = gr.Chatbot(label="Historical Bot", height=400)

    with gr.Row():
        msg = gr.Textbox(placeholder="e.g. Zusto (Salizzada, Ramo Salizzada)  a S. Giacomo dall’Orio.",
                         label="Your Place")
        submit_btn = gr.Button("📜 Ask about this place")


    def respond(place, history):
        answer = ask_about_place(place)
        history.append((place, answer))
        return history, place


    submit_btn.click(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, msg])

# Run the interface
if __name__ == "__main__":
    demo.launch()
