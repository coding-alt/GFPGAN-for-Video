import gradio as gr
from utils.gfpgan_wrapper import VideoEnhancer
import asyncio

video_enhancer = VideoEnhancer()

async def enhance_video(video, batch_size):
    print(f"Enhancing video: {video}")
    output_path = await video_enhancer.video_enhance(video, batch_size)
    print(f"Output path: {output_path}")
    return output_path

css = """footer {visibility: hidden}"""
with gr.Blocks(title="视频高清化", css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.sky)) as demo:
    gr.Markdown("# 视频高清化")
    gr.Markdown("上传一个视频以增强其质量。")
    
    with gr.Row():
        video_input = gr.Video(label="上传视频")
        video_output = gr.Video(label="高清化后的视频")
    
    with gr.Row():
        batch_size = gr.Slider(label="批量大小", minimum=1, maximum=256, value=128, step=1)
        btn = gr.Button("一键高清")

    btn.click(fn=enhance_video, inputs=[video_input, batch_size], outputs=video_output)

demo.launch(server_name="0.0.0.0")