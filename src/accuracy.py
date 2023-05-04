from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from acc_utils import LoadDatasetImg, LoadDatasetSkt
from torchvision.transforms import Compose, Resize, ToTensor


def get_acc(skt_model, img_model, batch_size=128, dataset='ClothesV1', mode='test', device='cuda'):
    print('Evaluating Network dataset [{}_{}] ...'.format(dataset, mode))

    data_set_skt = LoadDatasetSkt(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_set_img = LoadDatasetImg(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_loader_skt = DataLoader(data_set_skt, batch_size=batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)
    data_loader_img = DataLoader(data_set_img, batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)

    skt_model = skt_model.to(device)
    img_model = img_model.to(device)
    skt_model.eval()
    img_model.eval()

    top1_count = 0
    top5_count = 0
    top10_count = 0

    with torch.no_grad():
        Image_Feature = torch.FloatTensor().to(device)
        for imgs in tqdm(data_loader_img):
            img = imgs.to(device)
            img_feats = img_model(img)
            img_feats = F.normalize(img_feats, dim=1)
            Image_Feature = torch.cat((Image_Feature, img_feats.detach()))

        for idx, skts in enumerate(tqdm(data_loader_skt)):
            skt, skt_idx = skts
            skt, skt_idx = skt.to(device), skt_idx.to(device)
            skt_feats = skt_model(skt)
            skt_feats = F.normalize(skt_feats, dim=1)

            similarity_matrix = torch.argsort(torch.matmul(skt_feats, Image_Feature.T), dim=1, descending=True)

            top1_count += (similarity_matrix[:, 0] == skt_idx).sum()
            top5_count += (similarity_matrix[:, :5] == torch.unsqueeze(skt_idx, dim=1)).sum()
            top10_count += (similarity_matrix[:, :10] == torch.unsqueeze(skt_idx, dim=1)).sum()

        top1_accuracy = round(top1_count.item() / len(data_set_skt) * 100, 3)
        top5_accuracy = round(top5_count.item() / len(data_set_skt) * 100, 3)
        top10_accuracy = round(top10_count.item() / len(data_set_skt) * 100, 3)

    return top1_accuracy, top5_accuracy, top10_accuracy


def test_for_model():
    from backbone import EncoderViT

    checkpoint = torch.load('./checkpoint/0_Better_ChairV2/model_Best.pth')
    # checkpoint = torch.load('./checkpoint/0_Best_ChairV2/model_Best.pth')
    # checkpoint = torch.load('./checkpoint/0_Best_ClothesV1/model_Best.pth')
    print('Loading Pretrained model successful !'
          'Epoch:[{}]  |  Loss:[{}]'.format(checkpoint['epoch'], checkpoint['loss']))
    print('Top1: {} %  |  Top5: {} %  |  Top10: {} %'.format(checkpoint['top1'], checkpoint['top5'],
                                                             checkpoint['top10']))

    sketch_encoder = EncoderViT(num_classes=256, feature_dim=768,
                                encoder_backbone='vit_base_patch16_224')
    image_encoder = EncoderViT(num_classes=256, feature_dim=768,
                               encoder_backbone='vit_base_patch16_224')
    sketch_encoder.load_state_dict(checkpoint['skt_model'])
    image_encoder.load_state_dict(checkpoint['img_model'])

    # top1_accuracy, top5_accuracy, top10_accuracy = get_acc(sketch_encoder, image_encoder, batch_size=128,
    #                                                        dataset='ChairV2', mode='train')
    top1_accuracy, top5_accuracy, top10_accuracy = get_acc(sketch_encoder, image_encoder, batch_size=128,
                                                           dataset='ChairV2', mode='test')

    print('Top1: {:.3f} %  |  Top5: {:.3f} %  |  Top10: {:.3f} %'.format(top1_accuracy * 100,
                                                                         top5_accuracy * 100,
                                                                         top10_accuracy * 100))


if __name__ == '__main__':
    test_for_model()
