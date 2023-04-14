from dataset_chairv2 import *
from Net_Basic_V1 import Net_Basic
import time
import torch.optim as optim
import torch.utils.data as data
from evaluate_chairv2 import Evaluate_chairv2 as evaluate
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    model = Net_Basic()

    if opt.load_earlier:
        model.load_state_dict(torch.load('model_Best.pth'))
        print('model loaded')

    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    print('No. of params = ', get_n_params(model))
    print('No. of params = ', sum(p.data.nelement() for p in model.parameters() if p.requires_grad))

    dataset_sketchy_train = CreateDataset_Sketchy(opt)
    dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                               num_workers=int(opt.nThreads),
                                               collate_fn=dataset_sketchy_train.collate_self)

    top1_buffer = 0
    top5_buffer = 0
    top10_buffer = 0
    iter = 0

    for epoch in range(opt.niter):
        for i, sanpled_batch in enumerate(tqdm(dataloader_sketchy_train, 0)):
            model.train()
            iter += 1

            model_input = sanpled_batch['sketch_img'].to(device), \
                          [x.to(device) for x in sanpled_batch['sketch_boxes']], \
                          sanpled_batch['positive_img'].to(device), \
                          [x.to(device) for x in sanpled_batch['positive_boxes']], \
                          sanpled_batch['negetive_img'].to(device), \
                          [x.to(device) for x in sanpled_batch['negetive_boxes']]

            loss = model.Train(*model_input)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            top1, top5, top10 = evaluate(model, device)
            print('Epoch: {}   |   Top1: {}%   |   Top5: {}%   |   Top10: {}%'.format(epoch, top1, top5, top10))

            with open(f'./CMHM.txt', 'a', encoding='utf-8') as f:
                f.write('Epoch: {}   |   Top1: {}%   |   Top5: {}%   |   Top10: {}%'.format(epoch,
                                                                                            top1, top5,
                                                                                            top10) + '\n')
                f.close()

        if top1 > top1_buffer:
            torch.save(model.state_dict(), 'model_Best.pth')
            top1_buffer, top5_buffer, top10_buffer = top1, top5, top10
            print('Model Updated')

    print(top1_buffer, top5_buffer, top10_buffer)

    with open(f'./CMHM.txt', 'a', encoding='utf-8') as f:
        f.write('\nBest:  |   Top1: {}%   |   Top5: {}%   |   Top10: {}%'.format(top1_buffer, top5_buffer, top10_buffer))
        f.close()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = './ChairV2/ChairV2_Coordinate'
    opt.roor_dir = './ChairV2'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.batchsize = 12
    opt.nThreads = 2
    opt.lr = 0.0001
    opt.niter = 500
    opt.load_earlier = False
    main_train(opt)
