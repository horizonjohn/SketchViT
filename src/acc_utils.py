import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LoadDatasetSkt(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, transform):
        skt_list = []
        label_list = []
        img_name_list = os.listdir(img_folder_path)

        for img_name in img_name_list:
            skt_list_path = glob.glob(skt_folder_path + img_name.split('.')[0] + '_?.png')
            for skt_name in skt_list_path:
                skt_name = skt_name.replace("\\", "/").split('/')[-1].split('.')[0] + '.png'
                img_item = img_name_list.index(img_name)
                skt_list.append(skt_name)
                label_list.append(img_item)

        self.skt_folder_path = skt_folder_path
        self.transform = transform
        self.skt_list = skt_list
        self.label_list = label_list

    def __getitem__(self, item):
        skt_path = os.path.join(self.skt_folder_path, self.skt_list[item])
        sample_skt = self.transform((Image.fromarray(np.array(Image.open(skt_path).convert('RGB')))))
        image_idx = self.label_list[item]

        return sample_skt, image_idx

    def __len__(self):
        return len(self.skt_list)


class LoadDatasetImg(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, transform):
        self.transform = transform
        self.img_folder_path = img_folder_path
        self.img_list = os.listdir(img_folder_path)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_folder_path, self.img_list[item])
        sample_img = self.transform((Image.fromarray(np.array(Image.open(img_path).convert('RGB')))))

        return sample_img

    def __len__(self):
        return len(self.img_list)
