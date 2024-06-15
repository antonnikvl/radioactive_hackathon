import gradio as gr
import requests

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
    print(response)
    return response

def retrive_docs(message):
    return f"Нет документации на тему {message}"

def process(message, chat_history, system_prompt, corrector_prompt, temperature, max_tokens, model):
    if len(chat_history) == 0:
        documentation = retrive_docs(message)

        request = f"Вопрос пользователя:\n {message}. В ответе тебе можнт пригодиться следующая документация {documentation}"
        first_response = generate(system_prompt, [{"role": "user", "content": request}], temperature, max_tokens, model)
        messages = [{"role": "user", "content": request}, {"role": "assistant", "content": first_response}]

        correction_message = generate(corrector_prompt, [{"role": "user", "content": f"На запрос\n {message}\n с данными \n {documentation} \n был дан следующий ответ:\n {first_response}"}], temperature, max_tokens, model)
        messages.append({"role": "user",  "content": f"Исправь свой ответ на основе следующего отзыва:\n{correction_message}\n. Не упоминай сам отзыв и его части в обновленном ответе."})
        final_response = generate(system_prompt, messages, temperature, max_tokens, model)
        if final_response.find('CALL_OP') != -1:
            yield "К сожалению, я не могу отетить на ваш вопрос автоматически. Обращение передано оператору."
        else:
            yield final_response
    else:
        messages = []
        for request, response in chat_history:
            messages.append({"role": "user", "content": request})
            messages.append({"role": "assistant", "content": response})
        yield generate(system_prompt, messages + [{"role": "user", "content": message}], temperature, max_tokens, model)

if __name__ == "__main__":
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
            gr.Textbox("Ты - ассистент, дающий ответы на вопросы пользователя. Всегда отвечай на русском языке. Дополнительно вместо с вопросом пользователя тебе на вход будет по возможности представленна релевантная документация. Соблюдай формальный стиль общения. При невозможности дать ответ, сообщи, что обращение будет переданно оператору. В таком случае, закончи свой ответ текстом CALL_OP.", label="System prompt для генерации ответа"),
            gr.Textbox("Всегда отвечай на русском языке. Ты выполняешь роль критика и оцениваешь ответы другой модели от 0 до 5 по каждому из следующих критериев. Корректность и соблюдение формального стиля беседы. Совпадение темы ответа с темой запроса. Согласованность ответа с предоставленными данными из документации. Корректность вызова оператора или его отсутствия.", label="System prompt для критика"),
            gr.Slider(0, 1, 0.5, label="Температура", step=0.05),
            gr.Slider(128, 4096, 1024, label="Макс. токенов", step=32),
            gr.Dropdown(available_models, value=available_models[0], label="Модель"),
        ],
        additional_inputs_accordion=gr.Accordion("Дополнительные параметры параметры", open=False),
        examples=None,
    ).queue().launch()