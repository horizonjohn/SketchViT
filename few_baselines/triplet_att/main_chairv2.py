import torch.nn as nn
from dataset_chairv2 import *
import numpy as np
from Net_Basic_V1 import Net_Basic  # , Resnet_Network
from skt_net import SketchANetSBIR, SketchANetDSSA
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data as data
from evaluate_chairv2 import Evaluate_chairv2 as evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def main_train(opt):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('\n[INFO] Setting SEED: ' + str(seed))

    # model = Net_Basic()
    # model = SketchANetSBIR()
    model = SketchANetDSSA()

    model.to(device)
    model.train()

    dataset_sketchy_train = CreateDataset_Sketchy(opt)
    dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                               num_workers=int(opt.nThreads))

    Triplet_Criterion = nn.TripletMarginLoss(margin=0.3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(200 * len(dataloader_sketchy_train)))

    top1_buffer = 0
    top5_buffer = 0
    top10_buffer = 0
    iter = 0

    print('No. of parameters', get_n_params(model))

    for epoch in tqdm(range(opt.niter)):
        # for i, param_group in enumerate(optimizer.param_groups):
        #     print("Learning rate for parameter group {}: {}".format(i, param_group['lr']))

        for i, sanpled_batch in enumerate(dataloader_sketchy_train, 0):
            model.train()
            iter += 1
            sketch_anchor_embedding = model(sanpled_batch['sketch_img'].to(device))
            rgb_positive_embedding = model(sanpled_batch['positive_img'].to(device))
            rgb_negetive_embedding = model(sanpled_batch['negetive_img'].to(device))

            loss = Triplet_Criterion(sketch_anchor_embedding, rgb_positive_embedding, rgb_negetive_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        with torch.no_grad():
            top1, top5, top10 = evaluate(model)
            print('Epoch: {}   |   Top1: {}%   |   Top5: {}%   |   Top10: {}%'.format(epoch, top1, top5, top10))

            with open(f'./triplet_att.txt', 'a', encoding='utf-8') as f:
                f.write('Epoch: {}   |   Top1: {}%   |   Top5: {}%   |   Top10: {}%'.format(epoch,
                                                                                            top1, top5, top10) + '\n')
                f.close()

        if top1 > top1_buffer:
            torch.save(model.state_dict(), 'model_Best_Att.pth')
            top1_buffer, top5_buffer, top10_buffer = top1, top5, top10
            print('Model Updated')

    print(top1_buffer, top5_buffer, top10_buffer)

    with open(f'./triplet_att.txt', 'a', encoding='utf-8') as f:
        f.write('\nBest:  |   Top1: {}%   |   Top5: {}%   |   Top10: {}%'.format(top1_buffer, top5_buffer, top10_buffer))
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = 'ChairV2_Coordinate'
    opt.roor_dir = './ChairV2'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.batchsize = 8
    opt.nThreads = 2
    opt.lr = 0.0001
    opt.niter = 500
    opt.load_earlier = False
    print(opt)
    main_train(opt)
