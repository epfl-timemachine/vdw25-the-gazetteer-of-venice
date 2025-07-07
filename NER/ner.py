from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
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


def solve_paras(filepath):
    place_to_text = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    state = "looking_for_place"
    place_lines = []
    text_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == '&':
            if place_lines and text_lines:
                place = ' '.join([l.strip('%').strip() for l in place_lines]).strip()
                text = '\n'.join(text_lines).strip()
                place_to_text[place] = text
            # Reset for next block
            place_lines = []
            text_lines = []
            state = "looking_for_place"
        elif state == "looking_for_place":
            if stripped.startswith('%'):
                place_lines.append(stripped)
                state = "reading_place"
                if stripped.endswith('%'):
                    state = "reading_text"
        elif state == "reading_place":
            place_lines.append(stripped)
            if stripped.endswith('%'):
                state = "reading_text"
        elif state == "reading_text":
            text_lines.append(line.rstrip())

    with open("NER\\NER_results.txt", "w") as out:
        for place, in_text in place_to_text.items():
            entities = guess_agent.invoke({'Text Input': in_text}).content
            entities_split = entities.split('#')
            if place in entities_split:
                entities_split.remove(place)
                entities = '#'.join(entities_split).strip('#')
            print(f'%{place}%{entities}\n', file=out, flush=True)


base_system_msg = '''
You are a helpful assistant that performs entity recognition from a Venetian historical encyclopedia in Italian.
You are given a list of examples of owners of land register that are either institutions or people.
Your task is to identify the entities and provide me an entity list, with the entities separated by #.
Please also consider years and centuries as entities. Keep these in mind as the only numbers to identify. 
Other examples of entities are 'palazzo' and 'chiesa' and 'casa' and 'castel' (place descriptors).
In case of a compound entity, also extract the entities it includes separately.
Only output the list, nothing else. Include each distinct entity only once.
'''

few_shots_vals = [
    ("""Regale invero è il palazzo che qui sorge, fatto fabbricare dai Giustinian alla fine del secolo XIV,
oppure al principio del XV. Esso chiamavasi vo lgarmente il «palazzo delle due torri»,  per le due
torri che aveva sul comignolo.""", "Giustinian#palazzo delle due torri"),
    ("""Scorre sotto il ponte, ora detto «Pasqualigo»,  e prima «Noal»,  dalla patrizia famiglia Anovale, o
 Noale. Ciò è provato dalle «Genealogie»  di Marco Barbaro, e dalla «Cronolog. di famiglie Nob.
 Ven. in Candia» del Muazzo (Classe VII, Cod. 124 della Marciana) ove si legge: «Il Ponte di Noal
 in Sestier di Cannaregio, in contra’ di Santa Fosca, si chiama con quest o nome giacché fu fatto la
 prima volta da uno di questa casa Anoval» . Al certo troviamo il suddetto Ponte così denominato
 fino dal secolo XIII, esistendo nel «Codice del Piovego»  una sentenza del 1298 col titolo seguente:
 «Sententia Pontis de Noale qui de caetero debet reparari et refici per convicinos S. Fuscae,
 secundum antiquam consuetudinem».  Anche nel 1379, come è provato dall’elenco dei contribuenti
 prestiti per la guerra di Chioggia, i Noal abitavano a S. Fosca. Questa famiglia venne da Noale,
 castel lo nel Trivigiano, e fino dal 982 trovavasi fra noi. Fece parte nel 1212 delle cavallerie spedite
 nell’isola di Candia. Si estinse in Venezia nel 1583.""", "Pasqualigo#Noal#Anovale#Noale#«Genealogie»#Marco Barbaro#"
                                                           "«Cronolog. di famiglie Nob. Ven. in Candia»#Muazzo#,"
                                                           "Marciana#Candia#Il Ponte di Noal#Sestier di Cannaregio#"
                                                           "Santa Fosca#Anoval#casa#Ponte#XIII#"
                                                           "«Codice del Piovego»#1298#Pontis#Noale#1379#"
                                                           "S. Fuscae#Chioggia#S. Fosca#castel#Trivigiano#982#"
                                                           "1212#Venezia#1583"),
    ("""Non sappiamo perché questo Sottoportico, che r eca il N. A. 663, 
e che ricorda una famiglia Novello, altre volte qui domiciliata, manchi del suo nome scritto sulla muraglia, 
mentre lo troviamo nell’Anagrafi stampata per cura del Municipio nel 1841.""",
     "Sottoportico#663#Novello#Municipio#1841")
]

guesses_val = [{"Text Input": k, 'Output': v} for k, v in few_shots_vals]
guesses_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{Text Input}"),
        ("ai", "{Output}"),
    ]
)
guess_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=guesses_prompt,
    examples=guesses_val,
)

guess_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", base_system_msg),
        guess_few_shot_prompt,
        ("human", "{Text Input}"),
    ]
)

chatgpt_llm = get_client_with_different_temp()

guess_agent = guess_prompt | chatgpt_llm

solve_paras("out\\text_with_places_clean.txt")

# print(guess_agent.invoke({'Text Input': """Non sappiamo perché questo Sottoportico, che r eca il N. A. 663,
# e che ricorda una famiglia Novello, altre volte qui domiciliata, manchi del suo nome scritto sulla muraglia,
# mentre lo troviamo nell’Anagrafi stampata per cura del Municipio nel 1841."""}).content)
