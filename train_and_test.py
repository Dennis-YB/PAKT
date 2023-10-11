import pandas as pd
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import ResNet as ResNet
from torch.utils.data import Dataset
from dataset import SC_make_dataset, RC_make_dataset, BC_make_dataset
from util import cosine_similarity, roi_dataset, cal_metrics, multi_dataset
from model import similar_model, baselineresnet50
import torch.nn.functional as F
import torchvision.models as models
from thop import profile
import time

device = 'cuda:3'
epoch = 50000
class_num = 3
feature_num = 6
dataset_name = 'SC' # SC or RC or BC
image_format = 'tif' # SC:tif, RC and BC:png
train_group = '1'
test_group = '2'

if dataset_name == 'SC':
    make_dataset = SC_make_dataset
if dataset_name == 'BC':
    make_dataset = BC_make_dataset
elif dataset_name == 'RC':
    make_dataset = RC_make_dataset
train_dataset = make_dataset(train_group)
train_data = multi_dataset(train_dataset, dataset_name, image_format)
test_dataset = make_dataset(test_group)
test_data = multi_dataset(test_dataset, dataset_name, image_format)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
pretext_model = torch.load(r'./checkpoints/best_ckpt.pth', map_location=torch.device('cpu'))
# model = models.resnet50(pretrained=True)
model.fc = nn.Identity()
model.load_state_dict(pretext_model, strict=True)
### SSL baseline
# model = baselineresnet50(pretrained=True, progress=False, key="SwAV") # "BT","MoCoV2","SwAV"
model = model.to(device)
model.eval()

histo_path_1 = './dataset/'+dataset_name+'_histo_feature/histology1.npy' # 'BCC_morphology,           KIRC_large_nest'
histo_path_2 = './dataset/'+dataset_name+'_histo_feature/histology2.npy' # 'BCC_palisade,             KIRC_thin_wall_vessel'
histo_path_3 = './dataset/'+dataset_name+'_histo_feature/histology3.npy' # 'SCC_intercelluar_bridges, KIRP_foamy_macrophage'
histo_path_4 = './dataset/'+dataset_name+'_histo_feature/histology4.npy' # 'SCC_keratin_pearls,       KIRP_papillary'
histo_path_5 = './dataset/'+dataset_name+'_histo_feature/histology5.npy' # 'BD_in_situ,               KICH_clear_cell_membrane'
histo_path_6 = './dataset/'+dataset_name+'_histo_feature/histology6.npy' # 'BD_wideing,               KICH_flocculent'

train_similar_data = []
test_similar_data = []

print("### Start similar calculate ###")
with torch.no_grad():
    for batch in train_loader:
        histo_similar_list_1 = []
        histo_similar_list_2 = []
        histo_similar_list_3 = []
        histo_similar_list_4 = []
        histo_similar_list_5 = []
        histo_similar_list_6 = []

        ### calculate FLOPs and Params ###
        # flops, params = profile(model, (batch[0].to(device),))
        # print('Stage1, flops: %.2f M, params: %.2f M'%(flops / 1000000.0, params / 1000000.0))
        feature_scale1 = model(batch[0].to(device))
        feature_scale2 = model(batch[1].to(device))
        feature_scale3 = model(batch[2].to(device))
        feature_scale1 = feature_scale1.cpu().numpy()
        feature_scale2 = feature_scale2.cpu().numpy()
        feature_scale3 = feature_scale3.cpu().numpy()

        histo_feature_1 = np.load(histo_path_1)
        for case in histo_feature_1:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_1.append([simil1, simil2, simil3])
        histo_final_similar_1 = np.mean(histo_similar_list_1)

        histo_feature_2 = np.load(histo_path_2)
        for case in histo_feature_2:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_2.append([simil1, simil2, simil3])
        histo_final_similar_2 = np.mean(histo_similar_list_2)

        histo_feature_3 = np.load(histo_path_3)
        for case in histo_feature_3:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_3.append([simil1, simil2, simil3])
        histo_final_similar_3 = np.mean(histo_similar_list_3)

        histo_feature_4 = np.load(histo_path_4)
        for case in histo_feature_4:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_4.append([simil1, simil2, simil3])
        histo_final_similar_4 = np.mean(histo_similar_list_4)

        histo_feature_5 = np.load(histo_path_5)
        for case in histo_feature_5:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_5.append([simil1, simil2, simil3])
        histo_final_similar_5 = np.mean(histo_similar_list_5)

        histo_feature_6 = np.load(histo_path_6)
        for case in histo_feature_6:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_6.append([simil1, simil2, simil3])
        histo_final_similar_6 = np.mean(histo_similar_list_6)

        # print("Name is {}, Label is {}, simil is {},{},{},{},{},{},{}".format(batch[4], batch[3], BCC_m_final_similar, 
        # BCC_p_final_similar, SCC_ib_final_similar, SCC_kp_final_similar, BD_is_final_similar, BD_w_final_similar))
        train_similar_data.append([histo_final_similar_1, histo_final_similar_2, histo_final_similar_3, histo_final_similar_4,
        histo_final_similar_5, histo_final_similar_6, batch[3], batch[4]])

    distance_time_start = time.time()
    for batch in test_loader:
        histo_similar_list_1 = []
        histo_similar_list_2 = []
        histo_similar_list_3 = []
        histo_similar_list_4 = []
        histo_similar_list_5 = []
        histo_similar_list_6 = []

        feature_scale1 = model(batch[0].to(device))
        feature_scale2 = model(batch[1].to(device))
        feature_scale3 = model(batch[2].to(device))
        feature_scale1 = feature_scale1.cpu().numpy()
        feature_scale2 = feature_scale2.cpu().numpy()
        feature_scale3 = feature_scale3.cpu().numpy()

        histo_feature_1 = np.load(histo_path_1)
        for case in histo_feature_1:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_1.append([simil1, simil2, simil3])
        histo_final_similar_1 = np.mean(histo_similar_list_1)

        histo_feature_2 = np.load(histo_path_2)
        for case in histo_feature_2:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_2.append([simil1, simil2, simil3])
        histo_final_similar_2 = np.mean(histo_similar_list_2)

        histo_feature_3 = np.load(histo_path_3)
        for case in histo_feature_3:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_3.append([simil1, simil2, simil3])
        histo_final_similar_3 = np.mean(histo_similar_list_3)

        histo_feature_4 = np.load(histo_path_4)
        for case in histo_feature_4:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_4.append([simil1, simil2, simil3])
        histo_final_similar_4 = np.mean(histo_similar_list_4)

        histo_feature_5 = np.load(histo_path_5)
        for case in histo_feature_5:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_5.append([simil1, simil2, simil3])
        histo_final_similar_5 = np.mean(histo_similar_list_5)

        histo_feature_6 = np.load(histo_path_6)
        for case in histo_feature_6:
            simil1 = cosine_similarity(feature_scale1[0], case[0])
            simil2 = cosine_similarity(feature_scale2[0], case[0])
            simil3 = cosine_similarity(feature_scale3[0], case[0])
            histo_similar_list_6.append([simil1, simil2, simil3])
        histo_final_similar_6 = np.mean(histo_similar_list_6)

        test_similar_data.append([histo_final_similar_1, histo_final_similar_2, histo_final_similar_3, histo_final_similar_4,
        histo_final_similar_5, histo_final_similar_6, batch[3], batch[4]])
    distance_time_end = time.time()
    distance_time = distance_time_end - distance_time_start
        
print("### Finish similar calculate ###")
train_model = similar_model(feature_num, class_num)
train_model.to(device)
optimizer = torch.optim.Adam(train_model.parameters(),lr=0.01) # 0.01 is good for us
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
loss_func = torch.nn.CrossEntropyLoss()
torch.backends.cudnn.benchmark = True

print("### Start training ###")
top_acc = 0
for i in range(epoch):
    train_model.train()
    total_loss = 0
    true_predicted = 0
    for train_item in train_similar_data:
        similar_feature = torch.tensor([train_item[0:-2]], dtype=torch.float32)
        label = torch.tensor([train_item[-2]], dtype=torch.int64)
        similar_feature = similar_feature.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        predicted = train_model(similar_feature)
        loss = loss_func(predicted, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        predicted_label = torch.max(F.softmax(predicted[0], dim=0),dim=0)[1]
        if predicted_label == label:
                true_predicted += 1
    train_acc = true_predicted / len(train_similar_data) * 100
    epoch_loss = total_loss / len(train_similar_data)
    # scheduler.step()
    
    if (i+1)%500 == 0:
        print('------------------------------------------------------------------')
        print('{}/{} train loss: {:.4f}, train acc: {:.2f}%'.format((i+1), epoch, epoch_loss, train_acc))
        ##### start test #####
        infer_time_start = time.perf_counter()
        train_model.eval()
        test_true_predicted = 0
        confusion_matrix = np.zeros((class_num, class_num))
        for test_item in test_similar_data:
            test_similar_feature = torch.tensor([test_item[0:-2]], dtype=torch.float32)
            test_label = torch.tensor([test_item[-2]], dtype=torch.int64)
            test_similar_feature = test_similar_feature.to(device)
            test_label = test_label.to(device)
            test_predicted = train_model(test_similar_feature)
            test_predicted_label = torch.max(F.softmax(test_predicted[0], dim=0),dim=0)[1]
            if test_predicted_label == test_label:
                    test_true_predicted += 1
            # else:
            #         print(test_similar_feature, test_item[-1])
            confusion_matrix[test_label.cpu()][test_predicted_label.cpu()] += 1
        infer_time_end = time.perf_counter()
        infer_time = infer_time_end - infer_time_start
        all_time = (distance_time + infer_time) / len(test_similar_data)
        acc_test = test_true_predicted / len(test_similar_data) * 100
        # if acc_test > top_acc:
        #     top_acc = acc_test
        #     torch.save(train_model.state_dict(), f'./output/{dataset_name}_knowledge_best_paramters.pkl')
        matric = cal_metrics(confusion_matrix)
        print(dataset_name+' Test acc: {:.2f}%, time for one is {} s'.format(acc_test, all_time))
        print(dataset_name+' Test clinic_confusion_matrix: \n{}'.format(confusion_matrix))
        print(dataset_name+' Test P_R_Speci_F1: \n{}'.format(matric))