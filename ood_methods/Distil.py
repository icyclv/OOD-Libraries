from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
# import wandb
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

# wandb.init(
#     project="ood-loss",

#     config={
#     "T": 100,
#     "lr": 0.01,
#     "epoch": 10,
#     "architecture": "ResNet",
#     "dataset": "cifar10",
#     }
# )

def distil_eval(model, data_loader, logits, args, device):

    result = None

    for (images, labels) in tqdm(data_loader):
        images = images.to(device)
        distil_score = DISTIL(images, model, logits, args, device)

        if result is None:
            result = distil_score
        else:
            result = np.append(result, distil_score)

    return result


def DISTIL(origin_data, model, logits, args, device):
    score = []

    outputs = model(origin_data)
    p_labels = outputs.argmax(1)
    p_labels = p_labels.data.cpu()

    # get base logits
    base_logits = None
    for i in range(origin_data.shape[0]):

        if base_logits is None:
            base_logits = logits[p_labels[i]].unsqueeze(0)
        else:
            base_logits = torch.cat((base_logits, logits[p_labels[i]].unsqueeze(0)), dim=0)

    base_logits = base_logits.float().to(device)

    noise = torch.zeros(origin_data.size()[0], 3, 224, 224).to(device)
    
    for iters in range(50):
        noise.requires_grad_()

        tmp_pred = model.forward_noise(origin_data, noise)
        loss_batch, total_loss = kl_divergence(F.softmax(base_logits / args.T, dim=1), F.softmax(tmp_pred / args.T, dim=1))

        grad = torch.autograd.grad(total_loss, [noise])[0].detach()
        print(total_loss)

        noise = noise.detach() - 1 * grad

    noise = noise.detach().cpu()

    for idx in range(origin_data.shape[0]):
        print((torch.norm(noise[idx], p=2)).cpu())
        score.append(-(torch.norm(noise[idx], p=2)).cpu())

    return np.array(score)


def get_logits(model, data_loader, args, device, mode="train"):
    model.eval()
    result = [[] for i in range(args.num_classes)]

    with torch.no_grad():
        for (images, labels) in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            if mode == "train":
                output = output.data.cpu().numpy()
                for i in range(labels.size(0)):
                    result[labels[i]].append(output[i])
            else:
                p_labels = output.argmax(1)
                p_labels = p_labels.data.cpu().numpy()
                output = output.data.cpu().numpy()
                for i in range(labels.size(0)):
                    result[p_labels[i]].append(output[i])

    logits = []
    for i in range(args.num_classes):
        tmp = np.mean(result[i], axis=0)
        logits.append(tmp)
    
    return np.array(logits)


def kl_divergence(p, q):
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon

    kl_div_batch = torch.sum(p * torch.log(p / q), dim=1)
    kl_loss = torch.sum(kl_div_batch, dim=0)

    return kl_div_batch, kl_loss