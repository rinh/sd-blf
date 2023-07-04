import importlib
import importlib.util

# from inpaint_anything import lama_inpaint
from depends.inpaint_anything import lama_inpaint

# print(lama_inpaint.inpaint_img_with_lama)

exit()

import io
import base64
import json

import requests
import shared
from PIL import Image

import gradio as gr

from shared import base64_to_img


def sam_process(img, txt):
    with io.BytesIO() as output_bytes:
        img.save(output_bytes, format='PNG')

        payload = {
            'sam_model_name': 'sam_vit_h_4b8939.pth',
            'input_image': base64.b64encode(output_bytes.getvalue()).decode('utf-8'),
            'dino_enabled': True,
            'dino_model_name': 'GroundingDINO_SwinT_OGC (694MB)',
            'dino_text_prompt': txt,
            'dino_box_threshold': 0.3,
            'dino_preview_checkbox': False
        }
        response = requests.post(f"{shared.diffusion_api}/sam/sam-predict", json=payload)
        content_data = json.loads(response.content)
        print(content_data.keys())

        return [base64_to_img(i) for i in content_data['masks']]


def create_ui():
    demo = gr.Interface(
        fn=sam_process,
        inputs=["pil", "text"],
        outputs=gr.Gallery().style(preview=True)
    )

    demo.launch()


create_ui()
