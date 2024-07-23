import gradio as gr
from utils.gfpgan_wrapper import VideoEnhancer

video_enhancer = VideoEnhancer()
def enhance_video(video):
    print(f"Enhancing video: {video}")
    output_path = video_enhancer.video_enhance(video)
    print(f"Output path: {output_path}")
    return output_path

css = """footer {visibility: hidden}"""
with gr.Blocks(title="视频高清化", css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.sky)) as demo:
    gr.Markdown("# 视频高清化")
    gr.Markdown("上传一个视频以增强其质量。")
    
    with gr.Row():
        video_input = gr.Video(label="上传视频")
        video_output = gr.Video(label="高清化后的视频")
    
    btn = gr.Button("一键高清")
    btn.click(enhance_video, inputs=video_input, outputs=video_output)

demo.launch(server_name="0.0.0.0")