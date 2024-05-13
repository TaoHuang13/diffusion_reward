import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from termcolor import cprint
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()


# --------------------------------------------- #
#                  Data Utils
#            for video dataset
# --------------------------------------------- #

class VideoDataset(Dataset):
    def __init__(self, data_path, train=True, frames_per_sample=3, frame_skip=1, random_time=True, total_videos=-1):
        self.data_path = data_path 
        self.train = train
        self.frames_per_sample = frames_per_sample
        cprint(f"Frames per sample: {self.frames_per_sample}", "yellow")
        self.random_time = random_time
        cprint(f"Random time: {self.random_time}", "yellow")
        self.total_videos = total_videos
        self.frame_skip = frame_skip

        self.data_root = str(Path(__file__).parents[4]) + data_path
        self.images = []
        self.task_list = os.listdir(self.data_root)
        if 'clip_embs.npy' in self.task_list:
            self.task_list.remove('clip_embs.npy')
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
        image = (image / 127.5 - 1.0).astype(np.float32)
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
            assert i >= 0
            img_path = os.path.join(self.all_video_paths[video_index], f"{i}.png")
            img = Image.open(img_path)
            img = self.preprocess_image(img) # (3,64,64), [0,1]
            img = torch.tensor(img, dtype=torch.float32)
            prefinals.append(img)    

        data = torch.stack(prefinals)
        return data

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


class DMCVideoDataset(Dataset):

    def __init__(self, data_path, transform=None, train=True, frames_per_sample=5, frame_skip=4, random_time=True, random_horizontal_flip=True, color_jitter=0,
                 total_videos=-1, after_idx=False, size=None, task_name=None):

        self.data_path = data_path 
        self.train = train
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

        self.data_root =  data_path
        self.task_list = os.listdir(self.data_root)
        if task_name is not None:
            self.task_list = [task_name]
        cprint(f"Task list: {self.task_list}", "yellow")

        num_episodes = 10000 if self.train else 3
        self.videos = []
        self.labels = []
        task_npzs = []
        for task in self.task_list:
            task_path = os.path.join(self.data_root, task, 'train') if self.train else os.path.join(self.data_root, task, 'test')
            task_npzs.extend([os.path.join(task_path, video) for video in sorted(os.listdir(task_path))][:num_episodes])
            # self.labels.extend([int(task_id_map[task])] * num_episodes)

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
        # image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def window_stack(self, a, width=3, step=1):
        return torch.stack([a[i:1+i-width or None:step] for i in range(width)]).transpose(0, 1)

    def len_of_vid(self, index):
        video = self.videos[index]
        num_frames = len(video)
        return num_frames

    def __len__(self):
        return len(self.videos)

    def max_index(self):
        return len(self.videos)

    def __getitem__(self, index, time_idx=0):
        
        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = index % self.max_index()

        prefinals = []
        postfinals = []
        # labels = []
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
            assert i >= 0
            img = self.videos[video_index][i]
            img = self.preprocess_image(img) # (3,64,64), [0,1]
            img = torch.tensor(img, dtype=torch.float32)
            prefinals.append(img) 
        # labels.append(self.labels[video_index])      
            

            # #! assume skip=1
            # img_path = os.path.join(self.all_video_paths[video_index], f"{min(i+self.frames_per_sample, video_len-1)}.png")
            # img = Image.open(img_path)
            # img = self.preprocess_image(img) # (3,64,64), [0,1]
            # img = torch.tensor(img, dtype=torch.float32)
            # postfinals.append(img)        


        data = torch.stack(prefinals)
        # post_data = torch.stack(postfinals)
        #data = self.jitter(data) # (5,3,64,64), [0,1]

        if self.after_idx:
            return data, post_data
        else:
            return data#, torch.nn.functional.one_hot(torch.tensor(labels, dtype=int), num_classes=16)
            #return data,torch.tensor(labels, dtype=int)

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




class VideoDataLoader(DataLoader):
    def get_video(self, idx):
        return self.dataset.get_video(idx)


def load_video_data(cfg):
    if cfg.domain != 'dmc':
        train_data = VideoDataset(cfg.dataset_path, frames_per_sample=cfg.num_frames+1, frame_skip=cfg.frame_skip, train=True)
        train_loader = VideoDataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
        val_data = VideoDataset(cfg.dataset_path, frames_per_sample=cfg.num_frames+1, frame_skip=cfg.frame_skip, train=False)
        val_loader = VideoDataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)
    else:
        train_data = DMCVideoDataset(cfg.dataset_path, frames_per_sample=cfg.num_frames+1, frame_skip=cfg.frame_skip, train=True)
        train_loader = VideoDataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
        val_data = DMCVideoDataset(cfg.dataset_path, frames_per_sample=cfg.num_frames+1, frame_skip=cfg.frame_skip, train=False)
        val_loader = VideoDataLoader(val_data, batch_size=cfg.batch_size, shuffle=True)  

    return train_loader, val_loader