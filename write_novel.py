import os
import time
from huggingface_hub import InferenceClient

# Uses the GitHub secret token automatically
client = InferenceClient(
    model="Qwen/Qwen2.5-72B-Instruct",
    token=os.environ.get('HF_TOKEN')
)

if not os.path.exists('outline.txt'):
    print("Error: outline.txt file not found.")
    exit(1)

with open('outline.txt', 'r', encoding='utf-8') as f:
    outline = f.read()

completed = []
if os.path.exists('progress.txt'):
    with open('progress.txt', 'r', encoding='utf-8') as f:
        completed = f.read().splitlines()

chapters_generated_this_run = 0
MAX_CHAPTERS_PER_RUN = 2 # Prevents GitHub Action from timing out

for i in range(1, 35):
    if chapters_generated_this_run >= MAX_CHAPTERS_PER_RUN:
        print(f"Reached batch limit of {MAX_CHAPTERS_PER_RUN} chapters. Stopping to commit progress.")
        break

    chapter_file = f'chapter_{i}.txt'
    if str(i) in completed or os.path.exists(chapter_file):
        print(f'Chapter {i} already exists. Skipping.')
        continue

    print(f'\n=== Starting Generation for Chapter {i} ===')

    prev_chapter_context = ""
    if i > 1 and os.path.exists(f'chapter_{i-1}.txt'):
        with open(f'chapter_{i-1}.txt', 'r', encoding='utf-8') as f:
            prev_chapter_context = f.read()[-3000:]

    chapter_text = ""
    
    # Strictly broken into your 3 clean target bits
    bits = [
        "First Bit (First 800 words: Establishing the chapter setup, initial narrative hook, family/school context, and starting conversations)",
        "Second Bit (Middle 800 words: Moving into deep interactions, rising character tension, witty text-exchanges, and narrative progression)",
        "Third Bit (Final 800 words: Bringing the chapter events to a climax, wrapping up immediate scenes, and writing a strong emotional hook for the next chapter)"
    ]

    chapter_success = True

    for bit_index, bit_focus in enumerate(bits):
        print(f' -> Generating Chapter {i}, Part {bit_index + 1}/3 (800 words)...')
        
        prompt = (
            f"You are an expert novelist writing a Christian YA Romantic Comedy titled 'Text Me, Lord'.\n"
            f"Book Outline:\n{outline}\n\n"
            f"Previous Chapter Context:\n{prev_chapter_context}\n\n"
            f"Current Chapter Progress so far:\n{chapter_text}\n\n"
            f"TASK:\n"
            f"Write the exact NEXT part of Chapter {i}. Focus specifically on this segment: {bit_focus}.\n"
            f"CRITICAL REQUIREMENT:\n"
            f"You must write an immersive, extensive narrative section that is exactly 800 words long. "
            f"Maintain a warm, funny, heartfelt, and faith-filled tone with a witty teenage voice. "
            f"Write out complete dialogue and descriptive details. Do not summarize or skip time. "
            f"Pick up seamlessly from where the current progress ends."
        )

        bit_success = False
        for attempt in range(3):
            try:
                messages = [{'role': 'user', 'content': prompt}]
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=1500, 
                    temperature=0.75
                )
                bit_text = response.choices[0].message.content # Fixed payload index bug
                
                if chapter_text:
                    chapter_text += "\n\n" + bit_text.strip()
                else:
                    chapter_text = bit_text.strip()
                    
                bit_success = True
                time.sleep(20) # Safe cooldown for serverless tier stability
                break
            except Exception as e:
                print(f'    [!] API busy on Part {bit_index + 1} (Attempt {attempt+1}): {e}')
                time.sleep(35)

        if not bit_success:
            print(f"FAILED to generate Part {bit_index + 1} for Chapter {i}.")
            chapter_success = False
            break

    if chapter_success:
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(chapter_text)

        with open('progress.txt', 'a', encoding='utf-8') as f:
            f.write(f'{i}\n')

        print(f'✨ Successfully saved Chapter {i} (~2400 words total from 3 bits).')
        chapters_generated_this_run += 1
        time.sleep(10)
    else:
        print("Stopping script execution due to API connection failures.")
        exit(1)
        
