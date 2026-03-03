from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("models/LightOnOCR")

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Transcribe this Swedish document."}
]}]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
print(repr(text))