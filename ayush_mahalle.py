import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io

@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-AWQ", torch_dtype=torch.float16, device_map="cuda"
    )
    min_pixels = 256*28*28
    max_pixels = 1024*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor

def process_image(image, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open(io.BytesIO(image)),
                },
                {"type": "text", "text": "Your task is to extract the text from the provided image."},
            ],
        },
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

st.title("Qwen2VL Image Text Extraction")

model, processor = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Extract Text"):
        with st.spinner("Processing..."):
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            result = process_image(img_byte_arr, model, processor)
            st.subheader("Extracted Text:")
            for t in result.split("\n\n")[1:]:
                st.write(t)