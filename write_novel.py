import os
import time
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    token=os.environ['HF_TOKEN']
)

with open('outline.txt', 'r') as f:
    outline = f.read()

completed = []
if os.path.exists('progress.txt'):
    with open('progress.txt', 'r') as f:
        completed = f.read().splitlines()

for i in range(1, 35):
    chapter_file = f'chapter_{i}.txt'
    if str(i) in completed or os.path.exists(chapter_file):
        print(f'Chapter {i} already exists. Skipping.')
        continue

    print(f'Generating Chapter {i}...')

    prev_context = ""
    if i > 1 and os.path.exists(f'chapter_{i-1}.txt'):
        with open(f'chapter_{i-1}.txt', 'r') as f:
            prev_context = f.read()[-3000:]

    prompt = (
        f"You are an expert novelist writing a Christian YA Romantic Comedy titled 'Text Me, Lord'. "
        f"Based on this full outline:\n{outline}\n"
        f"Following these events from the previous chapter:\n{prev_context}\n\n"
        f"Write Chapter {i} of the novel. It must be a highly detailed, immersive narrative chapter, "
        f"aiming for approximately 2,400 words. Maintain a warm, funny, heartfelt, and faith-filled tone "
        f"with a witty teenage voice. Write out all dialogue, actions, family/school elements, and emotional progression. "
        f"Do not summarize."
    )

    success = False
    for attempt in range(3):
        try:
            messages = [{'role': 'user', 'content': prompt}]
            response = client.chat_completion(
                messages=messages,
                max_tokens=4000,
                temperature=0.7
            )
            output_text = response.choices[0].message.content

            with open(chapter_file, 'w') as f:
                f.write(output_text)

            with open('progress.txt', 'a') as f:
                f.write(f'{i}\n')

            print(f'Successfully wrote Chapter {i}.')
            time.sleep(8)
            success = True
            break
        except Exception as e:
            print(f'Error on Chapter {i} (Attempt {attempt+1}): {e}')
            time.sleep(20)

    if not success:
        print("Stopping workflow due to consecutive failures.")
        break
        
