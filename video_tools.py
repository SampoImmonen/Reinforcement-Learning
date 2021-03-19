import time
import os
import moviepy.video.io.ImageSequenceClip
from PIL import Image
import torch
import gym


def save_videofile(image_dir, video_name, fps=30):
    """
    Remember to adjust fps to your skipframe
    for example if skipframe=4, suitable fps is 15
    """
    image_files = [image_folder+'/'+img for img in sorted(os.listdir(image_folder), key=len) if img.endswith(".png")]
    print(image_files[:10])
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)


def show_episode(agent, env, save_video=False, video_path="movie.mp4", images_path="./video_images", fps=15):
    """
    agent is assumed to be a neural network
    """
    agent.cpu()
    agent.eval()

    is_done = False
    obs = env.reset()
    total_reward = 0
    frame = 0
    images = []
    while (not is_done):
        obs = obs.unsqueeze(0)
        if save_video:
            data = env.render(mode='rgb_array')
            img = Image.fromarray(data, 'RGB')
            images.append(img)
        else:
            env.render()
        obs, reward, is_done, _ = env.step(Agent.net(obs).max(1)[1].item())
        total_reward+=reward
        frame=+1
    print(f"total reward:{total_reward}")
    if save_video:
        for i, img in enumerate(images):
            img.save(images_path+"/image"+str(i)+".png")
        save_videofile(images_path, video_path, fps=fps)
        print(f"saved video as {video_path}")