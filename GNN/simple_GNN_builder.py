from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

with open("D:\\Facultate\\Venice Data Week\\openaikey.txt", 'r') as f:
    cont = f.readlines()

params = dict((v.replace('\n', '').split('=')) for v in cont)


def get_client_with_different_temp(temp: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        openai_organization=params['organization_id'],
        api_key=params['api_key'],
        model='gpt-4o',
        temperature=temp
    )


# System-only input
system_message = SystemMessage(content="You are a very skillful ML Engineer. Your task is to build a GNN-based network"
                                       "for a question answering task, which takes an entity graph as input (built"
                                       "with Networkx). "
                                       "The entities are text pieces -- so you also have to embed them -- "
                                       "and are flagged as either question or source, depending on their role. If they "
                                       "are labeled as question, it means they are part of the question. If they are "
                                       "labeled as source, it means they are part of the document sources to be used "
                                       "for answering. Edges are drawn between question entities and source entities "
                                       "where the question and the source are related, meaning that a particular "
                                       "source is part of the resources attached to a particular question."
                                       "Please only output the code, no other explanations included. Write the entire "
                                       "flow, from the graph input I described until the answer prediction. "
                                       "After that, I need to apply GNNExplainer on top of the result, so make "
                                       "sure this will be possible with the network you build. Keep in mind that a "
                                       "question might comprise multiple entities. Same for any source. Account for "
                                       "multiple questions and entities in your mock data, so that it matches my "
                                       "use-case. Within my use-case, I have to start from the entities belonging to a"
                                       "certain question and head towards the answer. Also perform an inference "
                                       "example at the end of training, where for a given question, the result "
                                       "(clear text) is returned.")

llm = get_client_with_different_temp()
# Invoke without user input
response = llm.invoke([system_message]).content
with open("GNN_code.txt", 'w') as f:
    print(response, file=f, flush=True)
