import pandas as pd
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import ResNet as ResNet
from torch.utils.data import Dataset
import csv
import torchvision.models as models
from model import baselineresnet50

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# SSL baseline
# mean = (0.70322989, 0.53606487, 0.66096631)
# std = (0.21716536, 0.26081574, 0.20723464)

trnsfrms_val = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)
class roi_dataset(Dataset):
    def __init__(self, img_csv,):
        super().__init__()
        self.transform = trnsfrms_val
        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst.filename[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image

device = 'cuda:0'

img_csv=pd.read_csv(r'./dataset/SC_histo_name/histology1.csv')
test_datat=roi_dataset(img_csv)
database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
pretext_model = torch.load(r'./checkpoints/best_ckpt.pth')
# model = models.resnet50(pretrained=True)
model.fc = nn.Identity()
model.load_state_dict(pretext_model, strict=True)
### SSL baseline
# model = baselineresnet50(pretrained=True, progress=False, key="SwAV") # "BT","MoCoV2","SwAV"

model = model.to(device)
model.eval()

feature_path = './dataset/SC_histo_feature/histology1'
feature_list = [] 
with torch.no_grad():
    for batch in database_loader:
        features = model(batch.to(device))
        features = features.cpu().numpy()
        feature_list.append(features)
    np.save(feature_path, feature_list)
    print('Finish one document !')
