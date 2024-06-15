import gradio as gr
import requests
import multiprocessing
import subprocess
import sys
import pandas as pd
import json
import re

debug = True

# load json from definitions.json
definitions = None
with open('src/definitions.json', 'r', encoding='utf-8') as f:
    definitions = json.load(f)
    definitions = {item['short']: item['def'] for item in definitions['list']}

def replace_definitions(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in definitions.keys()) + r')\b')
    def replacement(match):
        return definitions[match.group(0)]
    result = pattern.sub(replacement, text)
    return result

available_models = [
    "llama3-70b-iq2xxs",
    "mistral:7b-instruct-fp16",
    "phi3:3.8b-mini-128k-instruct-f16",
    "phi3:14b-medium-4k-instruct-q8_0",
    "llama3:8b-instruct-q8_0",
    "saiga3_fp16",
    "qwen2:7b-instruct-fp16",
]

# ollama endpoint
url = "http://127.0.0.1:11434/v1/chat/completions"

headers={ 'Content-type': 'application/json' }

# read three lines from config.txt and use as url + header params
try:
    with open("config.txt", "r") as f:
        url = f.readline().strip()
        headers["CF-Access-Client-Id"] = f.readline().strip()
        headers["CF-Access-Client-Secret"] = f.readline().strip()
except Exception as e:
    pass

def generate(system_prompt, dialog, temperature, max_tokens, model):
    messages = [{"role": "system", "content": system_prompt}]
    messages += dialog
    data = {
        "temperature": temperature,
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=data)
    response = response.json()['choices'][0]['message']['content']
    if debug:
        print("######################")
        print(response)
    return response


system_prompt = "Всегда отвечай на русском языке. Ты - ассистент, предоставляющий ответы на вопросы пользователя в формальном стиле. Тебе могут быть предоставлены дополнительные документы, содержащие релевантную информацию для ответа. Используй предоставленные данные максимально, по возможности цитируя их. Если предоставленная информация недостаточна или пользователь просит переколючить на человека, пиши только слово CALL_OP. Ты не можешь отвечать не на русском."

df = pd.read_csv("src/data/database.csv")

def retrive_docs(message):
    flask_url = "http://127.0.0.1:5000/predict"
    flask_headers={ 'Content-type': 'application/json' }
    data = {
        "Context": [message]
    }
    response = requests.post(flask_url, headers=flask_headers, json=data)
    docs = response.json()['TopDocumentIds']
    # retrieve docs from df using ids from docs list
    result = df.loc[df['id'].isin(docs)]['text'].tolist()
    # references = []
    # images = []
    # for d in docs:
    #     doc = df.loc[df['id'] == d]
    #     print(doc)
    return '\n\n'.join(result)

def process(message, chat_history, system_prompt=system_prompt, corrector_prompt="", temperature = 0.3, max_tokens=1024, model=available_models[0]):
    op_call_response = "К сожалению, я не могу отетить на ваш вопрос автоматически. Обращение передано оператору."
    message = replace_definitions(message)
    if len(chat_history) == 0:
        documentation = retrive_docs(message)
        if debug:
            print("*********************")
            print(documentation)

        request = f"Вопрос пользователя:\n{message}. В ответе тебе может пригодиться следующая документация\n{documentation}"
        first_response = generate(system_prompt, [{"role": "user", "content": request}], temperature, max_tokens, model)
        messages = [{"role": "user", "content": request}, {"role": "assistant", "content": first_response}]

        # correction_message = generate(corrector_prompt, [{"role": "user", "content": f"На запрос\n {message}\n с данными \n {documentation} \n был дан следующий ответ:\n {first_response}"}], temperature, max_tokens, model)
        # messages.append({"role": "user",  "content": f"Исправь свой ответ на основе следующего отзыва:\n{correction_message}\n. Не упоминай сам отзыв и его части в обновленном ответе."})
        # final_response = generate(system_prompt, messages, temperature, max_tokens, model)
        final_response = first_response
        if final_response.find('CALL_OP') != -1:
            print("CALL OPERATOR: ", message)
            yield op_call_response
        else:
            yield final_response
    else:
        messages = []
        for request, response in chat_history:
            messages.append({"role": "user", "content": request})
            messages.append({"role": "assistant", "content": response})
        final_response = generate(system_prompt, messages + [{"role": "user", "content": message}], temperature, max_tokens, model)
        if final_response.find('CALL_OP') != -1:
            print("CALL OPERATOR: ", message)
            yield op_call_response
        else:
            yield final_response

def run_retriever():
   subprocess.run([sys.executable, 'retriever/app.py'], cwd="src")

if __name__ == "__main__":
    app_process = multiprocessing.Process(target=run_retriever)
    app_process.start()
    
    gr.ChatInterface(
        process,
        title="Система тех. поддержки пользователей",
        description="Чат с ИИ-ассистентом",
        chatbot=gr.Chatbot(label="История переписки",),
        textbox=gr.Textbox(placeholder="Введите текст", container=False, scale=7),
        retry_btn="Сгенерировать ответ заново",
        clear_btn="Очистить",
        undo_btn=None,
        additional_inputs=[
            # gr.Textbox("Ты - ассистент, дающий ответы на вопросы пользователя. Всегда отвечай на русском языке. Дополнительно вместо с вопросом пользователя тебе на вход будет по возможности представленна релевантная документация. Соблюдай формальный стиль общения. При невозможности дать ответ, сообщи, что обращение будет переданно оператору. В таком случае, закончи свой ответ текстом CALL_OP.", label="System prompt для генерации ответа"),
            gr.Textbox(system_prompt, label="System prompt для генерации ответа"),
            gr.Textbox("Всегда отвечай на русском языке. Ты выполняешь роль критика и оцениваешь ответы другой модели от 0 до 5 по каждому из следующих критериев. Корректность и соблюдение формального стиля беседы. Совпадение темы ответа с темой запроса. Согласованность ответа с предоставленными данными из документации. Корректность вызова оператора или его отсутствия.", label="System prompt для критика"),
            gr.Slider(0, 1, 0.2, label="Температура", step=0.05),
            gr.Slider(128, 4096, 1024, label="Макс. токенов", step=32),
            gr.Dropdown(available_models, value=available_models[0], label="Модель"),
        ] if debug else [],
        additional_inputs_accordion=gr.Accordion("Дополнительные параметры параметры", open=False),
        examples=None,
    ).queue().launch()
    app_process.join()