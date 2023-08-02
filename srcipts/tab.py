import modules.scripts as scripts
import gradio as gr
import os

from modules import script_callbacks

from srcipts.run_process import run


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            input_image = gr.Image(label="Input image", elem_id="ia_input_image", source="upload", type="numpy", interactive=True)
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                value="Prompt",
            ),
            n_prompt = gr.Textbox(
                label="Negative prompt",
                lines=3,
                value="Negative prompt",
            ),
            btn = gr.Button(
                "Inpaint Image"
            ).style(
                full_width=False
            )
        with gr.Row():
            gallery = gr.Gallery(
                label="Outputs",
                show_label=False,
            )


        btn.click(
            dummy_images,
            inputs = [input_image, prompt, n_prompt],
            outputs = [gallery],
        )

        return [(ui_component, "Inpaint Image", "inpaint_image_tab")]

def dummy_images(input_image, prompt, n_prompt):
    return run(input_image, prompt, n_prompt)


script_callbacks.on_ui_tabs(on_ui_tabs)
