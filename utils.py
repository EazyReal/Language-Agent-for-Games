import os
import openai
import time


def write_to_file(directory, filename, content):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)
    with open(file_path, 'a') as file:
        file.write(content + '\n')


def ask(prompt, GPT_MODEL, MAX_TOKENS, LOG_PATH, F_LM):
    prompt_chat = [
        {"role": "user", "content": prompt.strip()},
    ]
    cnt = 0
    while True:
        try:
            if GPT_MODEL == 'gpt-3.5':
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=prompt_chat,
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                answer = response["choices"][0]["message"]['content'].strip()
                write_to_file(LOG_PATH, F_LM, 'Prompt: \n' + prompt +
                              '\nResponse: \n' + answer + '\n' + '='*20 + '\n')
                return answer
            else:
                raise Exception(f'no GPT_MODEL names {GPT_MODEL}')
        except openai.error.RateLimitError as e:
            retry_after = 3
            print(f"Rate limit error: {e}. Retrying in {retry_after} seconds.")
            time.sleep(retry_after)
        except openai.error.InvalidRequestError as e:
            # can try to eliminate some parts of the prompt to reduce the number of tokens
            raise Exception(f"Exceed max: {e}.")
        except openai.error.APIError:
            cnt += 1
            if cnt > 3:
                return 'APIError. Tried 3 times. Skip this one.'
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
