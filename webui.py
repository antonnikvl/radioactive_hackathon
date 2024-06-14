import gradio as gr
from threading import Thread

def process(message, chat_history, system_prompt, temperature, max_tokens):
    yield "Ответ на сообщение: " + message + "\n"

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
            gr.Textbox("Ты - ассистент, дающий ответы на вопросы пользователя.", label="Системный запрос"),
            gr.Slider(0, 1, 0.5, label="Температура", step=0.05),
            gr.Slider(128, 4096, 1024, label="Макс. токенов", step=32),
        ],
        additional_inputs_accordion=gr.Accordion("Дополнительные параметры параметры", open=False),
        examples=None,
    ).queue().launch()