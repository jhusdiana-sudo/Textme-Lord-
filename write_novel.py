import os
import time
from huggingface_hub import InferenceClient

# Utilizing standard serverless text-generation architecture to bypass the 402 chat router
client = InferenceClient(token=os.environ.get('HF_TOKEN'))
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

with open('outline.txt', 'r', encoding='utf-8') as f:
    outline = f.read()

completed = []
if os.path.exists('progress.txt'):
    with open('progress.txt', 'r', encoding='utf-8') as f:
        completed = f.read().splitlines()

chapters_done_this_run = 0

for i in range(1, 35):
    if chapters_done_this_run >= 2:
        print("Batch cap reached. Pausing execution.")
        break

    chapter_file = f'chapter_{i}.txt'
    if str(i) in completed or os.path.exists(chapter_file):
        print(f'Chapter {i} exists. Skipping.')
        continue

    print(f'\n=== Constructing Chapter {i} ===')
    
    prev_context = ""
    if i > 1 and os.path.exists(f'chapter_{i-1}.txt'):
        with open(f'chapter_{i-1}.txt', 'r', encoding='utf-8') as f:
            prev_context = f.read()[-2500:]

    chapter_text = ""
    bits = [
        "Part 1: Chapter opening setup, initial narrative hook, family context, and beginning dialogue (approx 800 words).",
        "Part 2: Core interaction, rising tension, and witty teenage conversations (approx 800 words).",
        "Part 3: Scene climax, wrapping up immediate events, and a hook for the next chapter (approx 800 words)."
    ]

    chapter_success = True

    for idx, bit_focus in enumerate(bits):
        print(f' -> Querying Bit {idx + 1}/3...')
        
        # Format explicitly as a raw text string context prompt to skip the chat engine restriction
        prompt = (
            f"<|im_start|>system\nYou are an expert novelist writing a Christian YA Romantic Comedy titled 'Text Me, Lord'. Write extensive narrative scenes with dialogue. Do not summarize.<|im_end|>\n"
            f"<|im_start|>user\nOutline:\n{outline}\n\nPrevious Chapter Context:\n{prev_context}\n\nCurrent Chapter Progress:\n{chapter_text}\n\nTask: Write the next 800 words focusing on: {bit_focus}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        bit_success = False
        for attempt in range(3):
            try:
                # Raw text generation task pipeline skips premium API endpoints completely
                response = client.text_generation(
                    prompt=prompt,
                    model=MODEL_ID,
                    max_new_tokens=1024,
                    temperature=0.7,
                    stop_sequences=["<|im_end|>"]
                )
                
                bit_text = response.strip().replace("<|im_end|>", "")
                
                if chapter_text:
                    chapter_text += "\n\n" + bit_text
                else:
                    chapter_text = bit_text
                    
                bit_success = True
                time.sleep(10)
                break
            except Exception as e:
                print(f'    [!] Worker busy (Attempt {attempt+1}): {e}')
                time.sleep(20)

        if not bit_success:
            chapter_success = False
            break

    if chapter_success:
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
        with open('progress.txt', 'a', encoding='utf-8') as f:
            f.write(f'{i}\n')
        print(f'Successfully built Chapter {i}.')
        chapters_done_this_run += 1
    else:
        print("Aborting run due to API errors.")
        exit(1)
        
