import base64
import imghdr
import io
import itertools
import json
import os
import time
from pathlib import Path

import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import arrow

import shared
from dotenv import load_dotenv

if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
load_dotenv()


def javascript_html():
    head = f'<script type="text/javascript" src="file=javascripts/dragdrop.js"></script>\n'

    return head


def reload_javascript():
    js = javascript_html()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def interrogate_info(img):
    payload = {
        "image": shared.img_to_base64(img),
        "model": "clip"
    }
    response = requests.post(url=f'{shared.diffusion_api}/sdapi/v1/interrogate', json=payload)
    prompt = json.loads(response.content).get("caption", "")
    return prompt


def get_new_wh(img, res="512"):
    # 获取图像的宽度和高度
    height, width = np.array(img).shape[:2]

    # 计算宽高比
    aspect_ratio = width / height

    if res == "4k":
        res_scale = 2160
    else:
        res_scale = 512

    # 计算等比例缩放后的最大尺寸
    if aspect_ratio > 1:
        new_width = res_scale
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = res_scale
        new_width = int(new_height * aspect_ratio)
    return new_width, new_height


def reisze_img(img):
    if img.width > 512 or img.height > 512:
        return img.resize(get_new_wh(img, res="512"))
    else:
        return img


counter = itertools.count(start=1)


def reload_sd_model(sd_model_checkpoint, **kwargs):
    # set option
    requests.post(url=f'{shared.diffusion_api}/sdapi/v1/options', json={
        "sd_model_checkpoint": sd_model_checkpoint
    })

    # reload
    requests.post(url=f'{shared.diffusion_api}/sdapi/v1/reload-checkpoint')


def save_to_output(img: Image, suffix: str):
    output_dir = Path(__file__).parent.absolute() / f"./output/{arrow.now().format('YYYY-MM-DD')}"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    x = "{:03d}".format(next(counter))
    filepath = output_dir / f"./{arrow.now().format('HHmm')}-{x}-{suffix}.png"
    img.save(filepath)


def generate_tile_img(img, prompt, neg_prompt):
    """
    使用 tile 增强细节
    :param img: PIL.Image
    :param prompt:
    :param neg_prompt:
    :return:
    """

    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "init_images": [shared.img_to_base64(img)],
        "steps": 25,
        "width": img.width * 2,
        "height": img.height * 2,
        "denoising_strength": 0.7,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        # "input_image": shared.img_to_base64(img),
                        "module": "tile_resample",
                        "model": "control_v11f1e_sd15_tile [a371b31b]",
                        "threshold_a": 4,  # 这个是用在 tile_resample 预处理的时候的 down sample rate
                        "control_mode": "Balanced"
                    }
                ]
            }
        }
    }
    response = requests.post(url=f'{shared.diffusion_api}/sdapi/v1/img2img', json=payload)
    content_data = json.loads(response.content)
    return content_data


def generate_canny_img(img, prompt, neg_prompt, **kwargs):
    """
    使用 canny 控制构图
    :param img: PIL.Image
    :param prompt:
    :param neg_prompt:
    :return:
    """
    new_width, new_height = get_new_wh(img)

    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "steps": 25,
        "width": new_width,
        "height": new_height,
        "enable_hr": True,
        "hr_scale": 2,
        "denoising_strength": 0.7,
        # "hr_upscaler": "Latent",
        # "hr_second_pass_steps": 20,
        # "hr_resize_x": new_width * 2,
        # "hr_resize_y": new_height * 2,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": shared.img_to_base64(img),
                        "module": "canny",
                        "model": "control_v11p_sd15_canny [d14c016b]"
                    }
                ]
            }
        }
    }
    payload.update(kwargs)
    response = requests.post(url=f'{shared.diffusion_api}/sdapi/v1/txt2img', json=payload)

    content_data = json.loads(response.content)
    return content_data


def generate_scribble_img(img, prompt, neg_prompt, **kwargs):
    """
    使用 canny 控制构图
    :param img: PIL.Image
    :param prompt:
    :param neg_prompt:
    :return:
    """
    new_width, new_height = get_new_wh(img)

    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "steps": 25,
        "width": new_width,
        "height": new_height,
        "enable_hr": True,
        "hr_scale": 2,
        "denoising_strength": 0.7,

        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": shared.img_to_base64(img),
                        "module": "scribble_hed",
                        "model": "control_v11p_sd15_scribble [d4ba51ff]"
                    }
                ]
            }
        }
    }
    payload.update(kwargs)

    response = requests.post(url=f'{shared.diffusion_api}/sdapi/v1/txt2img', json=payload)

    content_data = json.loads(response.content)
    return content_data


def generate_similar_img(img, tile_check, style_check, prompt_txt):
    """
    # 使用图片生成一张相似图
    # 使用获取到的内容
    # 1. interrogate 获取 clip 生成的 prompt
    # 2. 修改 prompt 符合 realition vision 的要求
    # 3. 添加 controlnet 的 canny
    # 4. 可选加强细节
    # 5. 输出结果

    :param img: PIL.Image
    :return:
    """
    img = reisze_img(img)

    shared.log.info(f"generating image...{img}")

    # 1
    if prompt_txt == "":
        prompt = interrogate_info(img)
        prompt = prompt.split(',')[0]
    else:
        prompt = prompt_txt

    # 2
    if style_check == "真实":
        new_prompt = f"RAW photo, {prompt}, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3  "
        new_neg_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        reload_sd_model("realisticVisionV20_v20")
    elif style_check == "绘本":
        new_prompt = f"(picture book, children's book, flat style:1.4) , {prompt}, (no_person , no_human , no_animal , no_creature:1.2) , (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality "
        new_neg_prompt = "(person, human,animal,creature:1.5),(mess,EasyNegative:1.5),(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        reload_sd_model("deliberate_v2")

    # 3
    if tile_check == "模仿" and style_check == "真实":
        content_data = generate_canny_img(img, new_prompt, new_neg_prompt)
        result_img = shared.base64_to_img(content_data['images'][0])
    elif tile_check == "模仿" and style_check == "绘本":
        content_data = generate_scribble_img(img, new_prompt, new_neg_prompt, sampler_name="UniPC")
        result_img = shared.base64_to_img(content_data['images'][0])
    elif tile_check == "复刻":
        content_data = generate_tile_img(img, new_prompt, new_neg_prompt)
        result_img = shared.base64_to_img(content_data['images'][0])

    # 5
    save_to_output(img, 'origin')
    save_to_output(result_img, 'result')
    info = f"""
分辨率：{result_img.width} * {result_img.height} <br />
提词：{prompt}
    """

    return [result_img], info


def clear_generate_mask(img, txt) -> np.ndarray:
    """
    使用 https://github.com/continue-revolution/sd-webui-segment-anything/blob/master/scripts/api.py
    返回 mask 的图
    :param img: PIL.Image
    :param txt:
    :return: np.ndarray
    """
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

        # 一共生成3张，选最后一张
        return shared.base64_to_img(content_data['masks'][-1])


def clear_generate_remove_anything_img(img, mask: np.ndarray):
    payload = {
        "image": shared.img_to_base64(img),
        "mask": base64.b64encode(mask).decode('utf8'),
        "shape": mask.shape
    }

    response = requests.post(url=f'{shared.inpaint_anything_api}/api/v1/rm_any', json=payload)
    content_data = json.loads(response.content)
    return shared.base64_to_img(content_data['image'])


def clear_anything_func(img, txt):
    """
    # 一键清图，调用自己的API
    :param img:
    :param txt:
    :return:
    """
    img = reisze_img(img)

    mask = clear_generate_mask(img, txt)
    mask_arr = np.array(mask)
    print(mask)
    print(mask_arr.shape)
    result_img = clear_generate_remove_anything_img(img, mask_arr)
    return result_img, mask


def wait_on_server():
    while 1:
        time.sleep(0.5)


def create_api(app):
    from api import Api
    api = Api(app)
    shared.log.info("loaded api.")
    return api


def webui():
    shared.log.info("Start SD-BLF")
    IS_DEBUG = os.getenv("DEBUG")
    shared.log.info(f"DEBUG Status: {IS_DEBUG}")

    reload_javascript()

    with gr.Blocks(title="BLF UX-AI Toolkit") as demo:
        gr.Markdown("""
        # BLF UX-AI Toolkit
        
        feature / issues @rinh
        
        """)

        with gr.TabItem(label="一键仿图"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Pil()
                    with gr.Accordion(label="高级选项", open=False):
                        input_prompt_txt = gr.Text(label="提词", placeholder="提词，如果为空则使用默认提词")
                    input_style_chk = gr.Radio(["真实", "绘本"],
                                               label="风格",
                                               info="真实-请提供一张真实的照片为原图;   绘本-请提供一张手绘草图为原图; ",
                                               value="真实")
                    input_tile_chk = gr.Radio(["模仿", "复刻"],
                                              label="生成方式",
                                              info="模仿-构图相同但内容不同;  复刻-几乎一致但细节增强",
                                              value="模仿")
                    btn = gr.Button(value='生成相似图')
                with gr.Column():
                    output_gly = gr.Gallery().style(preview=True)
                    # with gr.Row():
                    #     output_txt = gr.Text(label="自识别提词")
                    #     prompt_cp_btn = gr.Button(value="调整提词")
                    output_txt2 = gr.HTML(label="图片信息")

                btn.click(
                    fn=generate_similar_img,
                    inputs=[input_img, input_tile_chk, input_style_chk, input_prompt_txt],
                    outputs=[output_gly, output_txt2]
                )

                # prompt_cp_btn.click(
                #     fn=None,
                #     _js="(x) => x",
                #     inputs=output_txt,
                #     outputs=input_prompt_txt
                # )

        with gr.TabItem(label="一键清图"):
            with gr.Row():
                with gr.Column():
                    ra_input_img = gr.Pil()
                    ra_input_txt = gr.Text(label="要清除的内容", value="")
                    ra_btn = gr.Button(value='一键清图')
                with gr.Column():
                    ra_output_img = gr.Gallery().style(preview=True)

                ra_btn.click(
                    fn=clear_anything_func,
                    inputs=[ra_input_img, ra_input_txt],
                    outputs=[ra_output_img]
                )

            # send_btn.click(
            #     fn=None,
            #     _js=None,
            #     inputs=ra_output_img,
            #     outputs=input_img
            # )

    demo.queue(concurrency_count=64)

    app, local_url, share_url = demo.launch(server_name="0.0.0.0",
                                            server_port=7766,
                                            prevent_thread_lock=True,
                                            show_error=True,
                                            share=False if IS_DEBUG else True
                                            )
    create_api(app)

    wait_on_server()


if __name__ == '__main__':
    webui()
