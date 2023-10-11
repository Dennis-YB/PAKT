from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

transforms_val = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #imagenet
        # transforms.Normalize(mean = (0.70322989, 0.53606487, 0.66096631), std = (0.21716536, 0.26081574, 0.20723464))
    ]
)

def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)), 2)

class roi_dataset(Dataset):
    def __init__(self, img_list, dataset_name, image_format):
        super().__init__()
        self.transform = transforms_val
        self.images_lst = img_list
        self.dataset_name = dataset_name
        self.image_format = image_format

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        name = self.images_lst[idx][0]
        label = self.images_lst[idx][1]
        image_scale1 = Image.open(f'/home/yubo/dataset/{self.dataset_name}_slide_image/40X/{name}.{self.image_format}').convert('RGB')
        # image_scale2 = Image.open(f'/home/yubo/dataset/{self.dataset_name}_slide_image/20X/{name}.{self.image_format}').convert('RGB')
        # image_scale3 = Image.open(f'/home/yubo/dataset/{self.dataset_name}_slide_image/40X/{name}.{self.image_format}').convert('RGB')
        image_scale1 = self.transform(image_scale1)
        # image_scale2 = self.transform(image_scale2)
        # image_scale3 = self.transform(image_scale3)
        # return image_scale1, image_scale2, image_scale3, label, name
        return image_scale1, label, name

class multi_dataset(Dataset):
    def __init__(self, img_list, dataset_name, image_format):
        super().__init__()
        self.transform = transforms_val
        self.images_lst = img_list
        self.dataset_name = dataset_name
        self.image_format = image_format

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        name = self.images_lst[idx][0]
        label = self.images_lst[idx][1]
        image_scale1 = Image.open(f'/home/yubo/dataset/{self.dataset_name}_slide_image/10X/{name}.{self.image_format}').convert('RGB')
        image_scale2 = Image.open(f'/home/yubo/dataset/{self.dataset_name}_slide_image/20X/{name}.{self.image_format}').convert('RGB')
        image_scale3 = Image.open(f'/home/yubo/dataset/{self.dataset_name}_slide_image/40X/{name}.{self.image_format}').convert('RGB')
        image_scale1 = self.transform(image_scale1)
        image_scale2 = self.transform(image_scale2)
        image_scale3 = self.transform(image_scale3)
        return image_scale1, image_scale2, image_scale3, label, name

class single_patch_dataset(Dataset):
    def __init__(self, img_list, dataset_name, image_format):
        super().__init__()
        self.transform = transforms_val
        self.images_lst = img_list
        self.dataset_name = dataset_name
        self.image_format = image_format

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        name = self.images_lst[idx][0]
        label = self.images_lst[idx][1]
        img = Image.open(f'/home/yubo/dataset/{self.dataset_name}_slide_image/10X/{name}.{self.image_format}').convert('RGB')
        # fusion_img = self.transform(img)
        temp_array = np.array(img)
        row_mid = int(temp_array.shape[0]) // 2
        col_mid = int(temp_array.shape[1]) // 2
        patch1 = temp_array[0:row_mid,0:col_mid,:]
        patch1 = Image.fromarray(patch1)
        patch1 = self.transform(patch1).unsqueeze(0)
        patch2 = temp_array[0:row_mid,col_mid:,:]
        patch2 = Image.fromarray(patch2)
        patch2 = self.transform(patch2).unsqueeze(0)
        patch3 = temp_array[row_mid:,0:col_mid,:]
        patch3 = Image.fromarray(patch3)
        patch3 = self.transform(patch3).unsqueeze(0)
        patch4 = temp_array[row_mid:,col_mid:,:]
        patch4 = Image.fromarray(patch4)
        patch4 = self.transform(patch4).unsqueeze(0)
        fusion_img = torch.cat((patch1, patch2, patch3, patch4), 0)
        return fusion_img, label, name

def histo_dataset(img_list):
    histo_trait = []
    for index, row in img_list.iterrows():
        img = Image.open(row['filename']).convert('RGB')
        img_trans = transforms_val(img)
        histo_trait.append(img_trans)
    histo_trait = torch.stack(histo_trait, dim=0)
    return histo_trait

class split_slide_dataset(Dataset):
    def __init__(self, img_list, dataset_name, image_format):
        super().__init__()
        self.transform = transforms_val
        self.images_lst = img_list
        self.dataset_name = dataset_name
        self.image_format = image_format

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        name = self.images_lst[idx][0]
        label = self.images_lst[idx][1]
        origin_image = Image.open(f'/home/yubo/dataset/{self.dataset_name}_slide_image/10X/{name}.{self.image_format}').convert('RGB')
        temp_array = np.array(origin_image)
        row_mid = int(temp_array.shape[0]) // 2
        col_mid = int(temp_array.shape[1]) // 2
        patch1 = temp_array[0:row_mid,0:col_mid,:]
        patch1 = Image.fromarray(patch1)
        patch2 = temp_array[0:row_mid,col_mid:,:]
        patch2 = Image.fromarray(patch2)
        patch3 = temp_array[row_mid:,0:col_mid,:]
        patch3 = Image.fromarray(patch3)
        patch4 = temp_array[row_mid:,col_mid:,:]
        patch4 = Image.fromarray(patch4)
        image_scale0 = self.transform(origin_image)
        image_scale1 = self.transform(patch1)
        image_scale2 = self.transform(patch2)
        image_scale3 = self.transform(patch3)
        image_scale4 = self.transform(patch4)
        return image_scale0, image_scale1, image_scale2, image_scale3, image_scale4, label, name

def cal_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(confusion_matrix)
        # 对角线上是正确预测的
        TP = confusion_matrix[i, i]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(confusion_matrix[:, i]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(confusion_matrix[i, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN
        precision = round(TP/(TP+FP),4)
        recall = round(TP/(TP+FN),4)
        specificity = round(TN/(TN+FP),4)
        fscore = round((2*recall*precision)/(recall+precision),4)

        metrics_result.append([precision, recall, specificity, fscore])
    average = np.mean(metrics_result, axis=0)
    return average