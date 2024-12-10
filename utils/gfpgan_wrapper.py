import os
import cv2
import uuid
import re
import asyncio
from gfpgan import GFPGANer
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip

class GfpganWrapper:
    def __init__(self):
        # 加载模型
        CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(CURRENT_SCRIPT_PATH, '../gfpgan/weights', 'GFPGANv1.3' + '.pth')
        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None)
        self.restorer = restorer

    async def enhance_batch(self, images):
        tasks = [self.enhance(image) for image in images]
        return await asyncio.gather(*tasks)

    async def enhance(self, input_img):
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5)
        return restored_img

    async def video_enhance(self, video_path, batch_size):
        # 生成临时文件路径
        temp_dir = os.path.dirname(video_path)
        temp_file_name = str(uuid.uuid4())
        temp_img_dir = os.path.join(temp_dir, temp_file_name)
        os.makedirs(temp_img_dir, exist_ok=True)

        video_capture = cv2.VideoCapture(video_path)

        # 获取输入视频文件的基本信息
        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0

        # 循环读取视频中的每一帧
        while frame_count < total_frames:
            frames = []
            for _ in range(batch_size):
                if frame_count >= total_frames:
                    break
                success, image = video_capture.read()
                if not success:
                    break
                frames.append(image)
                frame_count += 1

            # 处理读取的帧
            if frames:
                enhanced_frames = await self.enhance_batch(frames)
                for i, frame in enumerate(enhanced_frames):
                    frame_path = os.path.join(temp_img_dir, f"{frame_count - len(frames) + i:08d}.png")
                    cv2.imwrite(frame_path, frame)
                print(f"Processed {frame_count} / {total_frames} frames")

        video_capture.release()

        def is_valid_image(file):
            pattern = re.compile(r'\d{8}\.png')
            return pattern.match(file)

        print('Start combining images...')
        files = [file for file in os.listdir(temp_img_dir) if is_valid_image(file)]
        files.sort(key=lambda x: int(x.split('.')[0]))
        img_list = [os.path.join(temp_img_dir, file) for file in files]

        tmp_video = os.path.join(temp_dir, f"{temp_file_name}.mp4")
        video_clip = ImageSequenceClip(img_list, fps=fps)
        video_clip.write_videofile(tmp_video, fps=fps, codec='libx264', audio=False)

        temp_video_clip = VideoFileClip(tmp_video)
        audio_clip = AudioFileClip(video_path)
        final_clip = temp_video_clip.set_audio(audio_clip)

        base_name, ext = os.path.splitext(os.path.basename(video_path))
        final_output_path = os.path.join(temp_dir, f"{base_name}_enhanced.mp4")
        final_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

        for file in files:
            os.remove(os.path.join(temp_img_dir, file))
        os.rmdir(temp_img_dir)
        os.remove(tmp_video)

        return final_output_path

    def image_enhance(self, image):
        print(f"Enhancing image: {image}")

        if not os.path.isfile(image):
            raise ValueError(f"Invalid image path: {image}")

        # 读取输入图片
        image_array = cv2.imread(image)

        # 对图片进行高清化处理
        enhanced_image = self.enhance(image_array)

        # 生成临时文件路径
        temp_dir = os.path.dirname(image)
        temp_file_name = str(uuid.uuid4()) + '.jpg'
        output_temp_path = os.path.join(temp_dir, temp_file_name)

        # 保存处理后的图片
        cv2.imwrite(output_temp_path, enhanced_image)

        print(f"Output path: {output_temp_path}")
        return output_temp_path

if __name__ == "__main__":
    video_path = "input_video_path.mp4"
    batch_size = 1

    video_enhancer = GfpganWrapper()
    enhanced_video_path = asyncio.run(video_enhancer.video_enhance(video_path, batch_size))
    print(f"Enhanced video saved to: {enhanced_video_path}")

    image_path = "input_image_path.jpg"
    enhanced_image_path = asyncio.run(video_enhancer.enhance_image(image_path))
    print(f"Enhanced image saved to: {enhanced_image_path}")
