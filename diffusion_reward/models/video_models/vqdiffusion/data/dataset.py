import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from termcolor import cprint
from torch.utils.data import Dataset
import cv2

class VideoDataset(Dataset):
    def __init__(self, data_root, phase='train', frames_per_sample=3, frame_skip=1, random_time=True, total_videos=-1):
        self.data_root = str(Path(__file__).parents[5]) + data_root

        self.train = phase == 'train'
        self.frames_per_sample = frames_per_sample
        cprint(f"Frames per sample: {self.frames_per_sample}", "yellow")
        self.random_time = random_time
        cprint(f"Random time: {self.random_time}", "yellow")
        self.total_videos = total_videos
        self.frame_skip = frame_skip

        self.task_list = os.listdir(self.data_root)
        cprint(f"Task list: {self.task_list}", "yellow")
        all_video_paths = []
        for task in self.task_list:
            task_path = os.path.join(self.data_root, task, 'train') if self.train else os.path.join(self.data_root, task, 'test')
            task_videos = [os.path.join(task_path, video) for video in os.listdir(task_path)]
            all_video_paths += task_videos
        self.all_video_paths = all_video_paths
        if self.total_videos > 0:
            # randomly select `total_videos` videos
            self.all_video_paths = np.random.choice(self.all_video_paths, self.total_videos)
        cprint(f"Total videos: {len(all_video_paths)}", "yellow")
        
        self.num_videos = len(all_video_paths)

    def preprocess_image(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = image.transpose(2, 0, 1)
        return image

    def window_stack(self, a, width=3, step=1):
        return torch.stack([a[i:1+i-width or None:step] for i in range(width)]).transpose(0, 1)

    def len_of_vid(self, index):
        video_path = self.all_video_paths[index % self.__len__()]
        num_frames = len(os.listdir(video_path))
        return num_frames

    def __len__(self):
        return len(self.all_video_paths) * 100 // self.frames_per_sample

    def max_index(self):
        return len(self.all_video_paths)

    def __getitem__(self, index, time_idx=0):
        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = index % self.max_index()
        prefinals = []
        video_len = self.len_of_vid(video_index)
        
        if self.random_time and video_len > 2 * (self.frames_per_sample - 1) * self.frame_skip:
            time_idx = np.random.choice(video_len)
        else:
            raise NotImplementedError

        if time_idx >= video_len / 2:
            frame_idxes = range(time_idx-(self.frames_per_sample-1)*self.frame_skip, time_idx+self.frame_skip, self.frame_skip)
        else:
            frame_idxes = range(time_idx, time_idx+self.frames_per_sample*self.frame_skip, self.frame_skip)
            
        for i in frame_idxes:
            img_path = os.path.join(self.all_video_paths[video_index], f"{i}.png")
            img = Image.open(img_path)
            img = self.preprocess_image(img) # (3,64,64), [0,1]
            img = torch.tensor(img, dtype=torch.float32)
            prefinals.append(img)        

        data = torch.stack(prefinals)
        item = {
            'image': data[-1],
            'frame': data[:-1]
        }
        return item

    def get_video(self, index):
        """
        get a sequence of full video
        """
        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = index % self.__len__()

        prefinals = []
        video_len = self.len_of_vid(video_index)
        
        for i in range(video_len):
            img_path = os.path.join(self.all_video_paths[video_index], f"{i}.png")
            img = Image.open(img_path)
            img = self.preprocess_image(img) # (3,64,64), [0,1]
            img = torch.tensor(img, dtype=torch.float32)
            prefinals.append(img)    

        data = torch.stack(prefinals)
        return data


class RealVideoDataset(Dataset):
    def __init__(self, data_root, phase='train', frames_per_sample=3, frame_skip=1, random_time=True, total_videos=-1):
        self.data_root = data_root

        self.train = phase == 'train'
        self.frames_per_sample = frames_per_sample
        cprint(f"Frames per sample: {self.frames_per_sample}", "yellow")
        self.random_time = random_time
        cprint(f"Random time: {self.random_time}", "yellow")
        self.total_videos = total_videos
        self.frame_skip = frame_skip

        self.task_list = os.listdir(self.data_root)
        cprint(f"Task list: {self.task_list}", "yellow")
        all_video_paths = []
        self.videos = []
        for task in self.task_list:
            task_path = os.path.join(self.data_root, task, 'train') if self.train else os.path.join(self.data_root, task, 'test')
            for video in os.listdir(task_path):
                for name in os.listdir(os.path.join(task_path, video)):
                    if '.pkl' in name:
                        all_video_paths.append(os.path.join(task_path, video, name))

        for pkl in all_video_paths:
            f = np.load(pkl, allow_pickle=True)
            self.videos.append(f['image'])

        cprint(f"Total videos: {len(self.videos)}", "yellow")
        self.num_videos = len(self.videos)

    def preprocess_image(self, image):
        image = np.array(image).astype(np.uint8)
        image = cv2.resize(image, (64, 64))
        #image = self.preprocessor(image=image)["image"]
        #image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def window_stack(self, a, width=3, step=1):
        return torch.stack([a[i:1+i-width or None:step] for i in range(width)]).transpose(0, 1)

    def len_of_vid(self, index):
        video = self.videos[index]
        num_frames = len(video)
        return num_frames

    def __len__(self):
        return len(self.videos)  * 100 // self.frames_per_sample

    def max_index(self):
        return len(self.videos)

    def __getitem__(self, index, time_idx=0):
        
        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = index % self.max_index()

        prefinals = []
        postfinals = []
        video_len = self.len_of_vid(video_index)
        
        # # sample a starting frame
        # if self.random_time and video_len > self.frames_per_sample:
        #     time_idx = np.random.choice(video_len - self.frames_per_sample)
        if self.random_time and video_len > 2 * (self.frames_per_sample - 1) * self.frame_skip:
            time_idx = np.random.choice(video_len)
        else:
            raise NotImplementedError

        if time_idx >= video_len / 2:
            frame_idxes = range(time_idx-(self.frames_per_sample-1)*self.frame_skip, time_idx+self.frame_skip, self.frame_skip)
        else:
            frame_idxes = range(time_idx, time_idx+self.frames_per_sample*self.frame_skip, self.frame_skip)
            
        for i in frame_idxes:
            img = self.videos[video_index][i]
            img = self.preprocess_image(img) # (3,64,64), [0,1]
            img = torch.tensor(img, dtype=torch.float32)
            prefinals.append(img)        

        data = torch.stack(prefinals)
        item = {
            'image': data[-1],
            'frame': data[:-1]
        }

        return item


class DMCVideoDataset(Dataset):
    def __init__(self, data_root, transform=None, phase='train', frames_per_sample=3, frame_skip=1, random_time=True, random_horizontal_flip=True, color_jitter=0,
                 total_videos=-1, after_idx=True, size=None, task=None):

        self.data_root = data_root

        self.train = phase == 'train'
        self.frames_per_sample = frames_per_sample
        cprint(f"Frames per sample: {self.frames_per_sample}", "yellow")
        self.random_time = random_time
        cprint(f"Random time: {self.random_time}", "yellow")
        self.random_horizontal_flip = random_horizontal_flip
        self.color_jitter = color_jitter
        self.total_videos = total_videos
        self.after_idx = after_idx
        self.size = size
        self.frame_skip = frame_skip

        self.task_list = os.listdir(self.data_root)
        if task is not None:
            self.task_list = [task]
        cprint(f"Task list: {self.task_list}", "yellow")

        num_episodes = 10000 if self.train else 3
        self.videos = []
        task_npzs = []
        for task in self.task_list:
            task_path = os.path.join(self.data_root, task, 'train') if self.train else os.path.join(self.data_root, task, 'test')
            task_npzs.extend([os.path.join(task_path, video) for video in sorted(os.listdir(task_path))][:num_episodes])

        for npz in task_npzs:
            f = np.load(npz)
            self.videos.append(f['image'])

        cprint(f"Total videos: {len(self.videos)}", "yellow")
        
        if self.train:
            #self.jitter = transforms.ColorJitter(hue=color_jitter)
            self.jitter = lambda x:x
        else:
            self.jitter = lambda x:x

        self.transform = transform
        # self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        # self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        # self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        self.num_videos = len(self.videos)

    def preprocess_image(self, image):
        image = np.array(image).astype(np.uint8)
        #image = self.preprocessor(image=image)["image"]
        #image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def window_stack(self, a, width=3, step=1):
        return torch.stack([a[i:1+i-width or None:step] for i in range(width)]).transpose(0, 1)

    def len_of_vid(self, index):
        video = self.videos[index]
        num_frames = len(video)
        return num_frames

    def __len__(self):
        return len(self.videos)  * 100 // self.frames_per_sample

    def max_index(self):
        return len(self.videos)

    def __getitem__(self, index, time_idx=0):
        
        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = index % self.max_index()

        prefinals = []
        postfinals = []
        video_len = self.len_of_vid(video_index)
        
        # # sample a starting frame
        # if self.random_time and video_len > self.frames_per_sample:
        #     time_idx = np.random.choice(video_len - self.frames_per_sample)
        if self.random_time and video_len > 2 * (self.frames_per_sample - 1) * self.frame_skip:
            time_idx = np.random.choice(video_len)
        else:
            raise NotImplementedError

        if time_idx >= video_len / 2:
            frame_idxes = range(time_idx-(self.frames_per_sample-1)*self.frame_skip, time_idx+self.frame_skip, self.frame_skip)
        else:
            frame_idxes = range(time_idx, time_idx+self.frames_per_sample*self.frame_skip, self.frame_skip)
            
        for i in frame_idxes:
            img = self.videos[video_index][i]
            img = self.preprocess_image(img) # (3,64,64), [0,1]
            img = torch.tensor(img, dtype=torch.float32)
            prefinals.append(img)        

        data = torch.stack(prefinals)
        item = {
            'image': data[-1],
            'frame': data[:-1]
        }

        return item


    def get_video(self, index):
        """
        get a sequence of full video
        """
        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = index % self.__len__()

        prefinals = []
        video_len = self.len_of_vid(video_index)
        
        for i in range(video_len):
            img = self.videos[video_index][i]
            img = self.preprocess_image(img) # (3,64,64), [0,1]
            img = torch.tensor(img, dtype=torch.float32)
            prefinals.append(img)    

        data = torch.stack(prefinals)
        # data = self.jitter(data) # (5,3,64,64), [0,1] # no jitter
       
        return data