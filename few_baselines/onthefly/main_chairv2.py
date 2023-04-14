from dataset_chairv2 import *
from Environment_SBIR import Environment
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn.utils as utils
import numpy as np

GAMMA = 0.9


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

    dataset_sketchy_train = CreateDataset_Sketchy(opt, on_Fly=True)
    dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                               num_workers=int(opt.nThreads))
    top1_buffer = 0
    top5_buffer = 0
    top10_buffer = 0
    mean_IOU_buffer = 0
    SBIR_Environment = Environment()
    loss_buffer = []

    optimizer = optim.Adam(SBIR_Environment.policy_network.parameters(), lr=opt.lr)

    step_stddev = 1
    SBIR_Environment.policy_network.train()

    for epoch in range(opt.niter):
        if mean_IOU_buffer > 0.25 and optimizer.param_groups[0]['lr'] == 0.001:
            optimizer.param_groups[0]['lr'] = 0.0001
            print('Reduce Learning Rate')

        print('LR value : {}'.format(optimizer.param_groups[0]['lr']))
        for i, sanpled_batch in enumerate(tqdm(SBIR_Environment.Sketch_Array_Train)):

            entropies = []
            log_probs = []
            rewards = []

            for i_sketch in range(sanpled_batch.shape[0]):
                action_mean, sketch_anchor_embedding, log_prob, entropy = \
                    SBIR_Environment.policy_network.select_action(sanpled_batch[i_sketch].unsqueeze(0).to(device))
                reward = SBIR_Environment.get_reward(sketch_anchor_embedding, SBIR_Environment.Sketch_Name_Train[i])

                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)

            loss_single = SBIR_Environment.calculate_loss(log_probs, rewards, entropies)
            loss_buffer.append(loss_single)

            step_stddev += 1

            # print('Epoch: {}, Iteration: {}, Loss: {}, REWARD: {}, Top1_Accuracy: {}, '
            #       'mean_IOU: {}, step: {}'.format(epoch, i, loss_single.item(),
            #                                       np.sum(rewards), top1_buffer, mean_IOU_buffer, step_stddev))

            if (i + 1) % 16 == 0:  # [Update after every 16 images]
                optimizer.zero_grad()
                policy_loss = torch.stack(loss_buffer).mean()
                policy_loss.backward()
                utils.clip_grad_norm_(SBIR_Environment.policy_network.parameters(), 40)
                optimizer.step()
                loss_buffer = []

        with torch.no_grad():
            top1, top5, top10, mean_IOU = SBIR_Environment.evaluate_RL(step_stddev)
            SBIR_Environment.policy_network.train()
            print('Epoch: {}, Top1: {}%, Top5: {}%, Top10: {}%'.format(epoch, top1, top5, top10))

            with open(f'./onthefly.txt', 'a', encoding='utf-8') as f:
                f.write('Epoch: {}   |   Top1: {}%   |   Top5: {}%   |   Top10: {}%'.format(epoch,
                                                                                            top1, top5, top10) + '\n')
                f.close()

        # print('Epoch: {}, Iteration: {}, Top1_Accuracy: {}, mean_IOU: {}'.format(epoch, i, top1, mean_IOU))

        if top1 > top1_buffer:
            mean_IOU_buffer = mean_IOU
            top1_buffer, top5_buffer, top10_buffer = top1, top5, top10
            torch.save(SBIR_Environment.policy_network.state_dict(), 'model_Best.pth')

            print('Model Updated')

    print(top1_buffer, top5_buffer, top10_buffer)

    with open(f'./onthefly.txt', 'a', encoding='utf-8') as f:
        f.write('\nBest:  |   Top1: {}%   |   Top5: {}%   |   Top10: {}%'.format(top1_buffer, top5_buffer, top10_buffer))
        f.close()

    # print(Top1_Song, meanIOU_Song)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = './ChairV2/ChairV2_Coordinate'
    opt.roor_dir = './ChairV2'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.batchsize = 1  # has to be one
    opt.nThreads = 2
    opt.lr = 0.001
    opt.niter = 1000
    opt.load_earlier = False
    main_train(opt)
