import argparse
import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import cv2
from sklearn import manifold
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_clothes_args_parser():
    parser = argparse.ArgumentParser(description='Load data path and save model path (ClothesV1)')
    parser.add_argument('--image_path_train', type=str,
                        default='./datasets/ClothesV1/trainB/', help='image datasets path (training)')
    parser.add_argument('--sketch_path_train', type=str,
                        default='./datasets/ClothesV1/trainA/', help='sketch datasets path (training)')
    parser.add_argument('--image_path_test', type=str,
                        default='./datasets/ClothesV1/testB/', help='image datasets path (testing)')
    parser.add_argument('--sketch_path_test', type=str,
                        default='./datasets/ClothesV1/testA/', help='sketch datasets path (testing)')
    parser.add_argument('--save_path', type=str,
                        default='./checkpoint/ClothesV1/', help='save model path')
    return parser


def get_chair_args_parser():
    parser = argparse.ArgumentParser(description='Load data path and save model path (ChairV2)')
    parser.add_argument('--image_path_train', type=str,
                        default='./datasets/ChairV2/trainB/', help='image datasets path (training)')
    parser.add_argument('--sketch_path_train', type=str,
                        default='./datasets/ChairV2/trainA/', help='sketch datasets path (training)')
    parser.add_argument('--image_path_test', type=str,
                        default='./datasets/ChairV2/testB/', help='image datasets path (testing)')
    parser.add_argument('--sketch_path_test', type=str,
                        default='./datasets/ChairV2/testA/', help='sketch datasets path (testing)')
    parser.add_argument('--save_path', type=str,
                        default='./checkpoint/ChairV2/', help='save model path')
    return parser


######################################
#         Neighbor Embedding         #
######################################

def umap_show(x, y):
    reducer = umap.UMAP(n_neighbors=6,
                        n_components=2,
                        min_dist=0.01,
                        metric='euclidean',
                        random_state=0)

    lenth = x.shape[0]
    data = np.append(x, y, axis=0)
    norm_dataset = reducer.fit_transform(data)
    X_norm = norm_dataset[:lenth, :]
    Y_norm = norm_dataset[lenth:, :]

    plt.figure()
    plt.scatter(X_norm[:, 0], X_norm[:, 1], color='#1E90FF', s=10, alpha=0.9, label='sketch')
    plt.scatter(Y_norm[:, 0], Y_norm[:, 1], color='#FF1493', s=10, alpha=0.9, label='image')

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def tsne_show(x, y):
    tsne = manifold.TSNE(n_components=2,
                         init='pca',
                         random_state=0)

    lenth = x.shape[0]
    data = np.append(x, y, axis=0)
    data_tsne = tsne.fit_transform(data)
    data_min, data_max = data_tsne.min(0), data_tsne.max(0)
    norm_dataset = (data_tsne - data_min) / (data_max - data_min)
    X_norm = norm_dataset[:lenth, :]
    Y_norm = norm_dataset[lenth:, :]

    plt.figure()
    plt.scatter(X_norm[:, 0], X_norm[:, 1], color='#1E90FF', s=10, alpha=0.9, label='sketch')
    plt.scatter(Y_norm[:, 0], Y_norm[:, 1], color='#FF1493', s=10, alpha=0.9, label='image')

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def visualization_embedding(dataset='test'):
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from datasets.load_dataset import LoadDatasetSkt, LoadDatasetImg
    from backbone import EncoderViT

    # get_parser = get_clothes_args_parser()
    get_parser = get_chair_args_parser()
    get_args = get_parser.parse_args()

    if dataset == 'train':
        skt_path = get_args.sketch_path_train
        img_path = get_args.image_path_train
    elif dataset == 'test':
        skt_path = get_args.sketch_path_test
        img_path = get_args.image_path_test
    else:
        raise ValueError("dataset must be 'train' or 'test' !")

    skt_set = LoadDatasetSkt(skt_path)
    img_set = LoadDatasetImg(img_path)
    skt_train_loader = DataLoader(skt_set, batch_size=16, shuffle=False,
                                  num_workers=2, pin_memory=True)
    img_train_loader = DataLoader(img_set, batch_size=16, shuffle=False,
                                  num_workers=2, pin_memory=True)

    skt_model = EncoderViT(num_classes=256, feature_dim=768,
                           encoder_backbone='vit_base_patch16_224')
    img_model = EncoderViT(num_classes=256, feature_dim=768,
                           encoder_backbone='vit_base_patch16_224')
    checkpoint = torch.load(get_args.save_path + 'model_Best.pth')
    print('Loading Pretrained model successful !'
          'Epoch:[{}]  |  Loss:[{}]'.format(checkpoint['epoch'], checkpoint['loss']))
    skt_model.load_state_dict(checkpoint['skt_model'])
    img_model.load_state_dict(checkpoint['img_model'])
    device = 'cuda'
    skt_model.to(device).eval()
    img_model.to(device).eval()

    img_feature = torch.FloatTensor().to(device)
    skt_feature = torch.FloatTensor().to(device)

    with torch.no_grad():
        for batch_idx, skt_anchor in enumerate(tqdm(skt_train_loader)):
            skt_anchor = skt_anchor.to(device)
            skt_feat = skt_model(skt_anchor)
            skt_feat = torch.nn.functional.normalize(skt_feat, dim=1)
            skt_feature = torch.cat((skt_feature, skt_feat.detach()))

        for batch_idx, img_anchor in enumerate(tqdm(img_train_loader)):
            img_anchor = img_anchor.to(device)
            img_feat = img_model(img_anchor)
            img_feat = torch.nn.functional.normalize(img_feat, dim=1)
            img_feature = torch.cat((img_feature, img_feat.detach()))

        similarity_matrix = torch.matmul(skt_feature, img_feature.T)
        similarity_matrix_idx = torch.argsort(similarity_matrix, dim=1, descending=True)
        # print(similarity_matrix)
        print(similarity_matrix_idx)

    tsne_show(skt_feature.detach().cpu().numpy(), img_feature.detach().cpu().numpy())
    # umap_show(skt_feature.detach().cpu().numpy(), img_feature.detach().cpu().numpy())


######################################
#              Grad_CAM              #
######################################

class ReshapeTransform:
    def __init__(self, model):
        input_size = model.encoder.patch_embed.img_size
        patch_size = model.encoder.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls_token, reshape to (H x W)
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # [B, H, W, C] -> [B, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def center_crop_img(img, size):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h + size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w + size]

    return img


def get_cam(img_path, model, target_layers, targets):
    data_transform = transforms.Compose([
        transforms.ToTensor()])

    # Prepare image
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    for i in range(len(target_layers)):
        plt.subplot(1, len(target_layers), i + 1)
        # Grad CAM
        cam = GradCAM(model=model, target_layers=target_layers[i], use_cuda=True,
                      reshape_transform=ReshapeTransform(model))
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        plt.xticks([])
        plt.yticks([])

    plt.show()


def attention_maps(dataset='skt'):
    from backbone import EncoderViT
    import os

    # get_parser = get_clothes_args_parser()
    get_parser = get_chair_args_parser()
    get_args = get_parser.parse_args()
    checkpoint = torch.load(get_args.save_path + 'model_Best_Acc.pth')
    print('Loading Pretrained model successful !'
          'Epoch:[{}]  |  Loss:[{}]'.format(checkpoint['epoch'], checkpoint['loss']))

    if dataset == 'skt':
        model = EncoderViT(num_classes=256, feature_dim=768, muti_heads=[8, 12, 16],
                           encoder_backbone='vit_base_patch16_224')
        model.load_state_dict(checkpoint['skt_model'])

        skt_dir = get_args.sketch_path_test
        skt_paths = os.listdir(skt_dir)

        for skt_path in skt_paths:
            image = os.path.join(skt_dir, skt_path)
            target_layers = [[model.encoder.blocks[-1].norm1],
                             [model.feat_block_1.norm1],
                             [model.feat_block_2.norm1],
                             [model.feat_block_3.norm1]]
            get_cam(image, model, target_layers, targets=None)

    elif dataset == 'img':
        model = EncoderViT(num_classes=256, feature_dim=768, muti_heads=[8, 12, 16],
                           encoder_backbone='vit_base_patch16_224')
        model.load_state_dict(checkpoint['img_model'])

        img_dir = get_args.image_path_test
        img_paths = os.listdir(img_dir)

        for img_path in img_paths:
            image = os.path.join(img_dir, img_path)
            target_layers = [[model.encoder.blocks[-1].norm1],
                             [model.feat_block_1.norm1],
                             [model.feat_block_2.norm1],
                             [model.feat_block_3.norm1]]
            get_cam(image, model, target_layers, targets=None)

    else:
        raise ValueError("dataset must be 'skt' or 'img' !")


if __name__ == '__main__':
    # visualization_embedding('test')
    attention_maps('img')
