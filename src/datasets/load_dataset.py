import os
import glob
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class LoadMyDataset(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, im_size=224):
        skt_list = []
        img_list = []
        # skt_pos_list = []
        img_name_list = os.listdir(img_folder_path)

        for pos_img_name in img_name_list:
            # 1.1 anchor_skt_name, pos_skt_name
            pos_skt_list_path = glob.glob(skt_folder_path + pos_img_name.split('.')[0] + '_?.png')
            pos_skt_list_name = [file_name.replace("\\", "/").split('/')[-1].split('.')[0] + '.png'
                                 for file_name in pos_skt_list_path]

            # 1.3 append lists
            if len(pos_skt_list_name) == 0:
                print(pos_img_name)

            for len_list in range(len(pos_skt_list_name)):
                skt_list.append(pos_skt_list_name[len_list])
                img_list.append(pos_img_name)

            # if len(pos_skt_list_name) == 0:
            #     continue
            #
            # elif len(pos_skt_list_name) == 1:
            #     skt_list.append(pos_skt_list_name[0])
            #     img_list.append(pos_img_name)
            #     skt_pos_list.append(pos_skt_list_name[0])
            #
            # else:
            #     random_idx = random.randint(1, len(pos_skt_list_name) - 1)
            #     skt_list.append(pos_skt_list_name[0])
            #     img_list.append(pos_img_name)
            #     skt_pos_list.append(pos_skt_list_name[random_idx])

        self.img_folder_path = img_folder_path
        self.skt_folder_path = skt_folder_path

        self.skt_list = skt_list
        self.img_list = img_list
        # self.skt_pos_list = skt_pos_list

        self.transform_anchor = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

        self.transform_aug = transforms.Compose([
            transforms.Resize((int(im_size * 1.2), int(im_size * 1.2))),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.CenterCrop((int(im_size), int(im_size))),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        # 1.1 anchor skt
        skt_path = os.path.join(self.skt_folder_path, self.skt_list[item])
        skt_anchor = self.transform_anchor(Image.fromarray(np.array(Image.open(skt_path).convert('RGB'))))
        skt_aug = self.transform_aug(Image.fromarray(np.array(Image.open(skt_path).convert('RGB'))))

        # 1.2 anchor img
        pos_path = os.path.join(self.img_folder_path, self.img_list[item])
        img_anchor = self.transform_anchor(Image.fromarray(np.array(Image.open(pos_path).convert('RGB'))))
        img_aug = self.transform_aug(Image.fromarray(np.array(Image.open(pos_path).convert('RGB'))))

        sample = skt_anchor, skt_aug, img_anchor, img_aug

        return sample

    def __len__(self):
        return len(self.skt_list)


class LoadDatasetSkt(Dataset):
    def __init__(self, skt_folder_path, im_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])
        self.skt_folder_path = skt_folder_path
        self.skt_list = os.listdir(skt_folder_path)

    def __getitem__(self, item):
        skt_path = os.path.join(self.skt_folder_path, self.skt_list[item])
        sample_skt = self.transform((Image.fromarray(np.array(Image.open(skt_path).convert('RGB')))))

        return sample_skt

    def __len__(self):
        return len(self.skt_list)


class LoadDatasetImg(Dataset):
    def __init__(self, img_folder_path, im_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])
        self.img_folder_path = img_folder_path
        self.img_list = os.listdir(img_folder_path)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_folder_path, self.img_list[item])
        sample_img = self.transform((Image.fromarray(np.array(Image.open(img_path).convert('RGB')))))

        return sample_img

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    from tqdm import tqdm
    import torchvision

    train_set = LoadMyDataset(img_folder_path='../datasets/ChairV2/trainB/',
                              skt_folder_path='../datasets/ChairV2/trainA/')

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    for batch_idx, data in enumerate(tqdm(train_loader)):
        skt_anchor, skt_aug, img_anchor, img_aug = data
        torchvision.utils.save_image(skt_anchor, '../data/{}_skt_anchor.png'.format(batch_idx))
        torchvision.utils.save_image(skt_aug, '../data/{}_skt_aug.png'.format(batch_idx))
        torchvision.utils.save_image(img_anchor, '../data/{}_img_anchor.png'.format(batch_idx))
        torchvision.utils.save_image(img_aug, '../data/{}_img_aug.png'.format(batch_idx))

    print('Train data:', len(train_set))
