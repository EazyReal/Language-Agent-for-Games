import os
import openai
import time
import re
from configs import LMConfig

def write_to_file(directory, filename, content):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    with open(file_path, 'a') as file:
        file.write(content + '\n')


def extract_enclosed_text(text: str, start_marker: str, end_marker: str) -> str:
    pattern = re.escape(start_marker) + "(.*?)" + re.escape(end_marker)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


def lm(prompt: str, config: LMConfig) -> str:
    prompt_chat = [
        {"role": "user", "content": prompt.strip()},
    ]
    cnt = 0
    while True:
        try:
            client = openai.OpenAI()
            completion = client.chat.completions.create(
                model=config.gpt_model,
                messages=prompt_chat,
                temperature=0,
                max_tokens=config.max_tokens,
                top_p=1,  # consider all top 100% tokens
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            answer = completion.choices[0].message.content.strip()
            write_to_file(config.log_path, config.log_file, 'Prompt: \n' + prompt +
                          '\nResponse: \n' + answer + '\n' + '='*20 + '\n')
            return answer
        except openai.RateLimitError as e:
            retry_after = 3
            print(f"Rate limit error: {e}. Retrying in {retry_after} seconds.")
            time.sleep(retry_after)
        except openai.APIError:
            cnt += 1
            if cnt > 3:
                raise Exception(f"APIError. Tried 3 times. Skip this one.")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
