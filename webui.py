import gradio as gr
from utils.gfpgan_wrapper import GfpganWrapper
import asyncio

gfpgan_wrapper = GfpganWrapper()

async def enhance_video(video, batch_size):
    print(f"Enhancing video: {video}")
    output_path = await gfpgan_wrapper.video_enhance(video, batch_size)
    print(f"Output path: {output_path}")
    return output_path

async def enhance_video(image):
    print(f"Enhancing image: {image}")
    output_path = await gfpgan_wrapper.image_enhance(image)
    print(f"Output path: {output_path}")
    return output_path

css = """footer {visibility: hidden}"""
with gr.Blocks(title="视频高清化", css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.sky)) as demo:
    gr.Markdown("# 视频和图片高清化")

    with gr.Tab("视频高清化"):
        gr.Markdown("上传一个视频以增强其质量。")
        with gr.Row():
            video_input = gr.Video(label="上传视频")
            video_output = gr.Video(label="高清化后的视频")
        btn_video = gr.Button("一键高清")
        btn_video.click(enhance_video, inputs=video_input, outputs=video_output)

    with gr.Tab("图片高清化"):
        gr.Markdown("上传一张图片以增强其质量。")
        with gr.Row():
            image_input = gr.Image(label="上传图片", type="filepath")
            image_output = gr.Image(label="高清化后的图片")
        btn_image = gr.Button("一键高清")
        btn_image.click(enhance_image, inputs=image_input, outputs=image_output)

demo.launch(server_name="0.0.0.0")
