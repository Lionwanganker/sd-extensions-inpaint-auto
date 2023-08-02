import base64
import io

import cv2
import numpy
import requests
from PIL import Image

def run(img_raw: Image, prompt, n_prompt):
    # A1111 URL
    url = "http://4.193.198.186:7861"

    # Read Image in RGB order
    img = cv2.cvtColor(numpy.asarray(img_raw),cv2.COLOR_RGB2BGR)

    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)

    encoded_image = base64.b64encode(bytes).decode('utf-8')

    playload = {
        "sam_model_name": "sam_vit_h_4b8939.pth",
        "input_image": encoded_image,
        "sam_positive_points": [],
        "sam_negative_points": [],
        "dino_enabled": True,
        "dino_model_name": "GroundingDINO_SwinT_OGC (694MB)",
        "dino_text_prompt": "front object",
        "dino_box_threshold": 0.3,
        "dino_preview_checkbox": False,
        "dino_preview_boxes_selection": [
            0
        ]
    }


    # Trigger Generation
    response = requests.post(url=f'{url}/sam/sam-predict', json=playload)

    # Read results
    r = response.json()
    result = r['masks'][2]
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    image.save('mask.png')

    playload = {
        "denoising_strength": 0.6,
        "prompt": prompt,
        # 提示词
        "negative_prompt": n_prompt,  # 反向提示词
        "mask": result,
        "mask_blur_x": 4,
        "mask_blur_y": 4,
        "inpainting_mask_invert": True,
        "init_images": [encoded_image],
        "seed": -1,  # 种子，随机数
        "batch_size": 5,  # 每次张数
        "n_iter": 1,  # 生成批次
        "steps": 35,  # 生成步数
        "cfg_scale": 10,  # 关键词相关性
        "width": 512,  # 宽度
        "height": 512,  # 高度
        "restore_faces": False,  # 脸部修复
        "tiling": False,  # 可平埔
        "inpainting_fill": 1, # 蒙版遮住的内容， 0填充， 1原图 2潜空间噪声 3潜空间数值零
        "inpaint_full_res": False,  # inpaint area, False: whole picture True：only masked
        "inpaint_full_res_padding": 32, # Only masked padding, pixels 32
        "override_settings": {
            "sd_model_checkpoint": "realisticVisionV50_v50VAE-inpainting.safetensors [f671072172]"
        },  # 一般用于修改本次的生成图片的stable diffusion 模型，用法需保持一致
        "sampler_index": "DPM++ 2M Karras",  # 采样方法
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": encoded_image,
                        "module": "inpaint_global_harmonious",
                        "model": "control_v11p_sd15_inpaint [ebff9138]",
                        "control_mode": "Balanced"
                    }
                ]
            }
        }
    }
    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=playload)
    # Read results
    r = response.json()
    print(r)
    result = r['images']
    rt = []
    i = 0
    for image in result:
        image = Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[0])))
        rt.append(image)
    return rt

if __name__ == '__main__':
    with Image.open("/Users/anker/Downloads/20230730-152349.jpeg") as img:
        imgArr = run(img,"((best quality)),((masterpiece)),(detailed),beach, bule sky, sea, sand, ocean, water, sun, sun shine, no human","human, man, woman")
        i = 0
        for image in imgArr:
            image.save(f'output{i}.png')
            i += 1