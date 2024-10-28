import requests
import re

API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
PARAMS = {
    "return_full_text": False,
    "max_new_tokens": 250
}

def query(payload):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def extract_generated_text(text):
    return text[0]['generated_text']

def extract_questions(text):
    text = extract_generated_text(text)
    questions = list(filter(lambda x: x, [a.split(':')[1] if re.search(r'Q\d:', a) else '' for a in text.split('\n')]))
    return questions

input = input("Input: ")

output = query({
	"inputs": f"""
    <|system|></s>
    <|user|>
    {input}</s>
    <|assistant|>""",
    "parameters": PARAMS
})

output_food_for_thought = query({
	"inputs": f"""
    <|system|></s>
    <|user|>
    Which are three questions Q1, Q2, and Q3 whose answers would be most helpful to solve the following problem: {input}</s>
    <|assistant|>""",
    "parameters": PARAMS
})

questions = extract_questions(output_food_for_thought)
question_answers = list()
for q in questions:
    answer = query({
        "inputs": f"""
        <|system|></s>
        <|user|>
        {q}</s>
        <|assistant|>""",
        "parameters": PARAMS
    })
    question_answers.append(extract_generated_text(answer))

final_query = f"<|system|></s>"

for i, q in enumerate(questions):
    final_query += f"""
        <|user|>
        {q}</s>
        <|assistant|>
        {question_answers[i]}</s>
    """

final_query += f"""
    <|user|>
    {input}</s>
    <|assistant|>"""

output_food_for_thought_final = query({
	"inputs": final_query,
    "parameters": PARAMS
})

print(extract_generated_text(output))
print("====")
print(extract_generated_text(output_food_for_thought_final))

# Input: List 2 thing that may be impacted by the use of AI in workplaces
#
# 1. Job displacement: One of the most significant impacts of AI in workplaces is the potential for job displacement. 
# As AI becomes more advanced and capable of performing tasks that were previously done by humans, some jobs may become obsolete. 
# For example, AI-powered chatbots and virtual assistants can handle customer service inquiries, freeing up human customer service 
# representatives to focus on more complex issues.
#
# 2. Increased productivity: Another impact of AI in workplaces is increased productivity. AI can automate repetitive and
# time-consuming tasks, allowing employees to focus on more strategic and creative work. For example, AI-powered software can#
# analyze large amounts of data and provide insights that would take humans hours or even days to uncover. This can lead to 
# faster decision-making and more efficient use of resources.
# ====
#
# 1. Job roles and responsibilities: As AI becomes more advanced, it may replace some human jobs, leading to job displacement
# and the emergence of new, higher-skill positions that require expertise in AI and related technologies. This can also lead to
# the augmentation of human capabilities, as AI can provide real-time insights, recommendations, and decision support,
# freeing up employees to focus on more complex and strategic work.
#
# 2. Workplace culture: The use of AI in workplaces can also impact workplace culture, as it may change the way people work,
# communicate, and collaborate. For example, AI can facilitate remote work and collaboration, as it allows employees to work
# from anywhere and communicate with each other in real-time. However, it can also lead to concerns about job security, privacy,
# and data protection, which can impact employee trust and engagement. Organizations will need to address these concerns and
# ensure that AI is used in a responsible and ethical manner, in order to maintain a positive and productive workplace culture.