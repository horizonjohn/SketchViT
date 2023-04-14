import torch
import torch.nn.functional as F
from Environment_SBIR import Environment
from tqdm import tqdm


def evaluate_RL(self, model):
    num_of_Sketch_Step = len(self.Sketch_Array_Test[0])  # [17, 2048]
    rank_all = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)  # [len(test), 17]
    for i_batch, sampled_batch in enumerate(tqdm(self.Sketch_Array_Test)):
        sketch_name = self.Sketch_Name_Test[i_batch]
        sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
        position_query = self.Image_Name_Test.index(sketch_query_name)

        for i_sketch in range(sampled_batch.shape[0]):  # [17, 2048] -->  [0, ..., 16]  # each sketch in 17 steps
            _, sketch_feature, _, _ = model.select_action(sampled_batch[i_sketch].unsqueeze(0).to(device))

            target_distance = F.pairwise_distance(F.normalize(sketch_feature),
                                                  self.Image_Array_Test[position_query].unsqueeze(0))
            distance = F.pairwise_distance(F.normalize(sketch_feature), self.Image_Array_Test)
            rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()  # distance <= target_distance --> True

    top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]  # 17th sketch True num
    top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]  # 17th sketch True num
    top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]  # 17th sketch True num

    return round(top1_accuracy * 100, 5), round(top5_accuracy * 100, 5), round(top10_accuracy * 100, 5)


def SBIR_evaluate():
    SBIR_Environment = Environment()
    model = SBIR_Environment.policy_network
    checkpoint = torch.load('./model_BestRL.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        top1, top5, top10 = evaluate_RL(SBIR_Environment, model)
        print('Top1: {}%, Top5: {}%, Top10: {}%, '.format(top1, top5, top10))


if __name__ == "__main__":
    device = torch.device('cuda')
    SBIR_evaluate()
