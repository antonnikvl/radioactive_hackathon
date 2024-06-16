# Atomic Hack 2 solution from (De)GenerativeTransformers team

# Установка
- Установить ollama (https://ollama.com/)
- Скачать llama3 70b iq2xxs (https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/blob/main/Meta-Llama-3-70B-Instruct-IQ2_XXS.gguf) в model
- В model выполнить ollama create llama3-70b-iq2xxs -f Modelfile
- установить python 3.10
- запустить install.bat или install.sh в зависимости от ос
- в случае ошибок можно сделать вручную:
 1. Создание venv: python3.10 -m venv ".venv"
 2. Активировать созданный venv
 3. Установить зависимости pip install -r requirements.txt
 4. Поправить баг в pdfminer. В pdfminer/image.py в _save_bytes обернуть Image.frombytes(...) ... img.save(fp) в try: ... except Exception: pass
- Разместить pdf в папку input/pdfs
- Запустить python prepare_index.py из директории src
- После этого можно запускать python webui.py в корневой директории проекта
- Открыть 127.0.0.1:7860/ в браузере