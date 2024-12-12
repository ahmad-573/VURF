from openai import OpenAI
import os

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Suggest some reasons why In-context learning to break down vision tasks into subtasks would work better than Visual Instruction Tuning? Give a short paragraph answer please"}
    ],
    temperature=0.7,
    max_tokens=512,
    top_p=0.5,
    frequency_penalty=0,
    presence_penalty=0,
    n=1
)
prog = completion.choices[0].message.content.lstrip('\n').rstrip('\n')
print(prog)