import pickle
import os
import argparse
import logging
import torch
import time
import scipy
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.data_processing as dp
import utils.hash_model as image_hash_model
import utils.calc_hr as calc_hr
import torch.nn as nn
import copy
import random
def load_label(label_filename, ind, DATA_DIR):
    label_filepath = os.path.join(DATA_DIR, label_filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    ind_filepath = os.path.join(DATA_DIR, ind)
    fp = open(ind_filepath, 'r')
    ind_list = [x.strip() for x in fp]
    fp.close()
    ind_np = np.asarray(ind_list, dtype=np.int)
    ind_np = ind_np - 1
    ind_label = label[ind_np, :]
    return torch.from_numpy(ind_label)



def GenerateCode(model_hash, data_loader, num_data, bit, k=0):
    B = np.zeros((num_data, bit), dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_img, _, data_ind = data
        data_img = Variable(data_img.cuda())
        if k == 0:
            _, out = model_hash(data_img)
            B[data_ind.numpy(), :] = torch.sign(out.data.cpu()).numpy()
    return B

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def Logtrick(x):
    lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt
def MLS3RDUH_algo(code_length):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(7)

    random.seed(10)
    DATA_DIR = '/data/home/trc/multi_label/COCO'
    LABEL_FILE = 'label_hot.txt'
    IMAGE_FILE = 'images_name.txt'
    DATABASE_FILE = 'database_ind.txt'
    TRAIN_FILE = 'train_ind.txt'
    TEST_FILE = 'test_ind.txt'
    top_k = 5000


    batch_size = 128
    epochs = 150
    learning_rate = 0.04 #0.05
    weight_decay = 10 ** -5
    data_set = 'alex_coco_final'
    net = 'alexnet_coco'
    model_name = 'alexnet'
    bit = code_length

    lamda = 0.001 #50cp
    print("*"*10, learning_rate, lamda, code_length, top_k, data_set, "*"*10)
    ### data processing



    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dset_database = dp.DatasetProcessingNUS_WIDE(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, DATABASE_FILE, transformations)

    dset_train = dp.DatasetProcessingNUS_WIDE(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, TRAIN_FILE, transformations)

    dset_test = dp.DatasetProcessingNUS_WIDE(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, TEST_FILE, transformations)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)
    database_loader = DataLoader(dset_database,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    test_loader = DataLoader(dset_test,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    train_labels = load_label(LABEL_FILE, TRAIN_FILE, DATA_DIR)
    database_labels = load_label(LABEL_FILE, DATABASE_FILE, DATA_DIR)
    test_labels = load_label(LABEL_FILE, TEST_FILE, DATA_DIR)
    label_size = test_labels.size()
    nclass = label_size[1]

    nnk = int(train_labels.size()[0] * 0.06)
    nno = int(train_labels.size()[0] * 0.06 * 1.5)

    hash_model = image_hash_model.HASH_Net(model_name, code_length)
    hash_model.cuda()
    optimizer_hash = optim.SGD(hash_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hash, step_size=500, gamma=0.5, last_epoch=-1)

    if os.path.exists(net + '_features4096_labels.pkl'):
        with open(net + '_features4096_labels.pkl', 'rb') as f:
            feature_label = pickle.load(f)
    else:
        x_train = torch.FloatTensor(num_train, 4096)
        label = torch.FloatTensor(num_train, nclass)
        for iter, traindata in enumerate(train_loader, 0):
            train_img, train_label, batch_ind = traindata
            train_img = Variable(train_img.cuda())
            feature_out, _ = hash_model(train_img)
            x_train[batch_ind, :] = feature_out.data.cpu()
            label[batch_ind, :] = train_label.type(torch.FloatTensor)

        feature_label = {'img_feature': x_train, 'label': label}
        with open(net + '_features4096_labels.pkl', 'wb') as f:
            pickle.dump(feature_label, f)

    feature_x = feature_label['img_feature']
    dim = feature_x.size()[0]
    normal = torch.sqrt((feature_x.pow(2)).sum(1)).view(-1, 1)
    normal_feature = feature_x / (normal.expand(dim, 4096))
    sim1 = normal_feature.mm(normal_feature.t())
    final_sim = sim1 * 1.0
    sim = sim1 - torch.eye(dim).type(torch.FloatTensor)
    top = torch.rand((1, dim)).type(torch.FloatTensor)
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = top.sort()[1][0]
        zero = torch.zeros(dim).type(torch.FloatTensor)
        zero[top20[-nnk:]] = 1.0
        sim[i, :] = top[0, :] * zero

    A = (sim > 0.0001).type(torch.FloatTensor)
    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)
    aa = dim - (sum_row > 0).sum()
    kk = sum_row.sort()[1]
    res_ind = list(range(dim))
    for ind in range(aa):
        res_ind.remove(kk[ind])
    res_ind = random.sample(res_ind, dim - aa)
    ind_to_new_id = {}
    for i in range(dim - aa):
        ind_to_new_id[i] = res_ind[i]
    res_ind = (torch.from_numpy(np.asarray(res_ind))).type(torch.LongTensor)
    sim = sim[res_ind, :]
    sim = sim[:, res_ind]
    sim20 = {}
    dim = dim - aa
    top = torch.rand((1, dim)).type(torch.FloatTensor)
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = top.sort()[1][0]
        zero = torch.zeros(dim).type(torch.FloatTensor)
        zero[top20[-nnk:]] = 1.0
        k = list(top20[-nnk:])
        sim20[i] = k
        sim[i, :] = top[0, :] * zero
    A = (sim > 0.0001).type(torch.FloatTensor)

    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)

    sum_row = sum_row.pow(-0.5)
    sim = torch.diag(sum_row)
    A = A.mm(sim)
    A = sim.mm(A)
    alpha = 0.99
    manifold_sim = (1 - alpha) * torch.inverse(torch.eye(dim).type(torch.FloatTensor) - alpha * A)

    manifold20 = {}
    for i in range(dim):
        top[0, :] = manifold_sim[i, :]
        top20 = top.sort()[1][0]
        k = list(top20[-nno:])
        manifold20[i] = k
    for i in range(len(sim20)):
        aa = len(manifold20[i])
        zz = copy.deepcopy(manifold20[i])
        ddd = []
        for k in range(aa):
            if zz[k] in sim20[i]:
                sim20[i].remove(zz[k])
                manifold20[i].remove(zz[k])
                ddd.append(ind_to_new_id[zz[k]])
        j = ind_to_new_id[i]
        for l in ddd:
            final_sim[j, l] = 1.0
        for l in sim20[i]:
            final_sim[j, ind_to_new_id[l]] = 0.0


    # final_sim = ((final_sim + final_sim.t()) > 0.1).type(torch.FloatTensor) - ((final_sim + final_sim.t()) < -0.1).type(torch.FloatTensor)
    f1 = (final_sim > 0.999).type(torch.FloatTensor)
    f1 = ((f1 + f1.t()) > 0.999).type(torch.FloatTensor)
    f2 = (final_sim < 0.0001).type(torch.FloatTensor)
    f2 = ((f2 + f2.t()) > 0.999).type(torch.FloatTensor)
    final_sim = final_sim * (1. - f2)
    final_sim = final_sim * (1. - f1) + f1
    final_sim = 2 * final_sim - 1.0

    for epoch in range(epochs):
        scheduler.step()
        epoch_loss = 0.0
        epoch_loss_r = 0.0
        epoch_loss_e = 0.0
        for iter, traindata in enumerate(train_loader, 0):
            train_img, train_label, batch_ind = traindata
            train_img = Variable(train_img.cuda())
            S = final_sim[batch_ind, :]
            S = S[:, batch_ind]
            the_batch = len(batch_ind)
            _, hash_out = hash_model(train_img)
            loss_all = (torch.log(torch.cosh((hash_out.mm(hash_out.t()) / float(code_length) \
                                              - Variable(S.cuda()))))).sum() / (the_batch * the_batch)
            Bbatch = torch.sign(hash_out)
            regterm = (Bbatch - hash_out).pow(2).sum() / (the_batch * the_batch)
            optimizer_hash.zero_grad()
            loss_all.backward()
            optimizer_hash.step()
            epoch_loss += loss_all.data[0]
            epoch_loss_r += regterm.data[0]
        print('[Train Phase][Epoch: %3d/%3d][Loss_i: %3.5f, Loss_e: %3.5f, Loss_r: %3.5f]' %
              (epoch + 1, epochs, epoch_loss / len(train_loader), epoch_loss_e / len(train_loader),
               epoch_loss_r / len(train_loader)))

    torch.save(hash_model.state_dict(), './final_codes/' + str(code_length) + 'bit_' + data_set + '_image_model.pkl')
    hash_model.eval()
    qi = GenerateCode(hash_model, test_loader, num_test, bit)
    ri = GenerateCode(hash_model, database_loader, num_database, bit)

    map_5000 = calc_hr.calc_topMap(qi, ri, test_labels.numpy(), database_labels.numpy(), top_k)
    print('map_', top_k, ':', map_5000)



if __name__ == "__main__":
    bits = [64]
    for i in range(len(bits)):
        print(50 * '*', bits[i], 50 * '*')
        MLS3RDUH_algo(bits[i])
