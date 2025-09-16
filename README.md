# bcc-ai-decentrathon

По всем вопросам пишите в тг @dias_kh

Основным инструментом для работы с OCR является [DotsOCR](https://github.com/dots-ocr/dots-ocr).

## Установка зависимостей

Установите необходимые зависимости:

- VLLM
- Ollama

## Настройка и запуск

### 1. Скачивание модели

```bash
python3 dots.ocr/tools/download_model.py
export hf_model_path=./weights/DotsOCR
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
```

**Важно:** Используйте имя директории без точек (например, `DotsOCR` вместо `dots.ocr`) для пути сохранения модели.

### 2. Настройка VLLM

```bash
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' `which vllm`
```

### 3. Запуск VLLM сервера

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --chat-template-content-format string --served-model-name model --trust-remote-code --max-model-len 64000
```

**Примечание:** Если возникает ошибка `ModuleNotFoundError: No module named 'DotsOCR'`, проверьте имя директории модели.

### 4. Тестирование VLLM API

```bash
python3 ./demo/demo_vllm.py --prompt_mode prompt_layout_all_en
```

## Парсинг документов

После запуска VLLM сервера можно парсить изображения или PDF файлы:

```bash
python3 dots_ocr/parser.py demo/demo_pdf1.pdf --num_thread 64
```

**Совет:** Используйте больше потоков (`--num_thread`) для PDF с большим количеством страниц.

## Демо

Запустите демо интерфейс:

```bash
python3 dots_ocr/demo/demo_gradio.py
```
