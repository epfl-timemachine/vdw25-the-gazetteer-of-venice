import pickle
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

threshold = 0.5
base_system_msg = ("Given a place and some entities, build a short explanatory text description of the place, using"
                   "the entities. The place is preceded by the keyword 'Place' and the entities are listed"
                   "after the keyword 'Entities' and separated by the special character #.")

with open("D:\\Facultate\\Venice Data Week\\openaikey.txt", 'r') as f:
    cont = f.readlines()

params = dict((v.replace('\n', '').split('=')) for v in cont)
with open("mapped_names.pkl", "rb") as f:
    loaded_entities = pickle.load(f)
    loaded_entities = {k: v for d in loaded_entities for k, v in d.items()}
    print(loaded_entities)


def get_client_with_different_temp(temp: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        openai_organization=params['organization_id'],
        api_key=params['api_key'],
        model='gpt-4o',
        temperature=temp
    )


# returned from the GNN training: explainer
def ask_about_place(place):
    if place not in loaded_entities:
        return "Sorry, no data found for the specified place."

    # most_connected_entities = retrieved from explainer, over threshold
    most_connected_entities = "#".join(loaded_entities[place]).strip("#")

    # Give place and most_connected_entities to Chatbot
    # instructions = (f"Build an explanatory short description of the place: {place}, by using these entities: "
    #                 f"{most_connected_entities}")

    guess_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_system_msg),
            ("human", "{Text Input}"),
        ]
    )

    chatgpt_llm = get_client_with_different_temp()

    guess_agent = guess_prompt | chatgpt_llm
    in_text = f"Place:{place}\n Entities:{most_connected_entities}"
    return guess_agent.invoke({'Text Input': in_text}).content


# print(ask_about_place('Zusto (Salizzada, Ramo Salizzada)  a S. Giacomo dall’Orio.'))
demo = gr.Interface(
    fn=ask_about_place,
    inputs=gr.Textbox(label="Enter Place Name"),
    outputs=gr.Textbox(label="Relevant Description"),
    title="Chatbot -- Historical Place Explainer",
    description="Enter a known place name to generate an explanatory text based on connected entities."
)

if __name__ == "__main__":
    demo.launch()
