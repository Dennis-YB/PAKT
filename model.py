import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import ResNet as RetcclResNet
from torchvision.models.resnet import Bottleneck, ResNet

class similar_model(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(similar_model, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(torch.ones(n_feature, n_class)))

    def forward(self, similar_feature):
        result = torch.mm(similar_feature, self.weight)
        return result

class DDKD(torch.nn.Module):
    def __init__(self, histo_num, n_class):
        super(DDKD, self).__init__()
        self.feature_extract = RetcclResNet.resnet50(num_classes=n_class, mlp=False, two_branch=False, normlinear=True)
        self.feature_extract.fc = nn.Identity()
        self.feature_extract.load_state_dict(torch.load('./checkpoints/best_ckpt.pth', map_location=torch.device('cpu')), strict=True)
        # self.histo_classify = nn.Linear(2048, histo_num)
        ### SSL baseline
        # self.feature_extract = baselineresnet50(pretrained=True, progress=False, key="BT") # "BT","MoCoV2","SwAV"
        self.weight = nn.Parameter(torch.empty(histo_num, n_class))
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

        for param in self.feature_extract.parameters():
            param.requires_grad = False

    def forward(self,img_1,img_2,img_3,histo_1,histo_2,histo_3,histo_4,histo_5,histo_6):
        our_trait_1 = self.feature_extract(img_1)
        our_trait_2 = self.feature_extract(img_2)
        our_trait_3 = self.feature_extract(img_3)
        # histo_pre = self.histo_classify(our_trait)
        histo_t_1 = self.feature_extract(histo_1)
        histo_t_2 = self.feature_extract(histo_2)
        histo_t_3 = self.feature_extract(histo_3)
        histo_t_4 = self.feature_extract(histo_4)
        histo_t_5 = self.feature_extract(histo_5)
        histo_t_6 = self.feature_extract(histo_6)
        # all_our_trait = torch.cat((our_trait_1, our_trait_2, our_trait_3),0)
        # all_histo_trait = torch.cat((histo_t_1,histo_t_2,histo_t_3,histo_t_4,histo_t_5,histo_t_6),0)
        # scale_similarity = torch.cat((torch.cosine_similarity(our_trait, histo_t_1),torch.cosine_similarity(our_trait, histo_t_2),torch.cosine_similarity(our_trait, histo_t_3),
        # torch.cosine_similarity(our_trait, histo_t_4),torch.cosine_similarity(our_trait, histo_t_5),torch.cosine_similarity(our_trait, histo_t_6)),0)
        similarity_1 = torch.mean(torch.cat((torch.cosine_similarity(our_trait_1, histo_t_1),torch.cosine_similarity(our_trait_2, histo_t_1),torch.cosine_similarity(our_trait_3, histo_t_1))))
        similarity_2 = torch.mean(torch.cat((torch.cosine_similarity(our_trait_1, histo_t_2),torch.cosine_similarity(our_trait_2, histo_t_2),torch.cosine_similarity(our_trait_3, histo_t_2))))
        similarity_3 = torch.mean(torch.cat((torch.cosine_similarity(our_trait_1, histo_t_3),torch.cosine_similarity(our_trait_2, histo_t_3),torch.cosine_similarity(our_trait_3, histo_t_3))))
        similarity_4 = torch.mean(torch.cat((torch.cosine_similarity(our_trait_1, histo_t_4),torch.cosine_similarity(our_trait_2, histo_t_4),torch.cosine_similarity(our_trait_3, histo_t_4))))
        similarity_5 = torch.mean(torch.cat((torch.cosine_similarity(our_trait_1, histo_t_5),torch.cosine_similarity(our_trait_2, histo_t_5),torch.cosine_similarity(our_trait_3, histo_t_5))))
        similarity_6 = torch.mean(torch.cat((torch.cosine_similarity(our_trait_1, histo_t_6),torch.cosine_similarity(our_trait_2, histo_t_6),torch.cosine_similarity(our_trait_3, histo_t_6))))
        # similarity_2 = torch.mean(torch.cosine_similarity(our_trait, histo_t_2))
        # similarity_3 = torch.mean(torch.cosine_similarity(our_trait, histo_t_3))
        # similarity_4 = torch.mean(torch.cosine_similarity(our_trait, histo_t_4))
        # similarity_5 = torch.mean(torch.cosine_similarity(our_trait, histo_t_5))
        # similarity_6 = torch.mean(torch.cosine_similarity(our_trait, histo_t_6))
        similarity = torch.stack([similarity_1, similarity_2, similarity_3, similarity_4, similarity_5, similarity_6],dim=0)
        # multi_similarity = torch.cosine_similarity(all_our_trait, all_histo_trait)
        # sort, _ = torch.sort(similarity, descending=True)
        # histo_pseudo = (similarity>=sort[1]).unsqueeze(0)
        result = torch.mm(similarity.unsqueeze(0), self.weight)
        return result

model = similar_model(6,4)
simi = torch.from_numpy(np.array([[0.5,0.5,0.5,0.5,0.5,0.5]]))
simi = simi.to(torch.float32)
output = model(simi)

class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(1, -1)
        return x


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def baselineresnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        # print(verbose)
    return model