import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import torch
from torch.optim import lr_scheduler
import json
from opt import opt
from data import Data
from network_resnet import SimpleResNet
from loss_res import ResnetLoss
from utils.get_optimizer import get_optimizer
from utils.extract_feature_vit import extract_feature
from utils.metrics import re_ranking
import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):

            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            class_scores, features = self.model(inputs)
            loss = self.loss(class_scores, features, labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def evaluate(self, epoch):
        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()
        epoch_json = 'metric/metric_epoch' + str(epoch)
        # epoch_json = 'metric/metric_epoch_450'
        os.makedirs(epoch_json)

        def result(distmat, query_ids=None, gallery_ids=None,
                    query_cams=None, gallery_cams=None, title = None):
            m, n = distmat.shape
            # Fill up default values
            if query_ids is None:
                query_ids = np.arange(m)
            if gallery_ids is None:
                gallery_ids = np.arange(n)
            if query_cams is None:
                query_cams = np.zeros(m).astype(np.int32)
            if gallery_cams is None:
                gallery_cams = np.ones(n).astype(np.int32)
            # Ensure numpy array
            query_ids = np.asarray(query_ids)
            gallery_ids = np.asarray(gallery_ids)
            query_cams = np.asarray(query_cams)
            gallery_cams = np.asarray(gallery_cams)
            # Sort and find correct matches
            indices = np.argsort(distmat, axis=1)

            dd = []
            for i in range(m):
                # Filter out the same id and same camera
                d = {}
                d['query_id'] = query_ids[i].astype(np.int32).tolist()
                valid = ((gallery_ids[indices[i]] != query_ids[i]) &
                         (gallery_cams[indices[i]] != query_cams[i]))
                ans_ids = gallery_ids[indices[i]][valid]
                d['ans_ids'] = ans_ids.astype(np.int32).tolist()
                dd.append(d)
            with open(epoch_json +'/'+ title + '.json', 'w', encoding='utf-8') as json_file:
                json.dump(dd, json_file, ensure_ascii=False)
            print('json finished')


        ######################### no re-rank ##########################
        dist = cdist(qf, gf)
        result(dist, self.queryset.ids, self.testset.ids, title = 'without_rerank')

        ######################### re-rank ##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        result(dist, self.queryset.ids, self.testset.ids, title='rerank')

        ######################### query expansion ##########################
        qf_new = []
        T = 9
        for i in range(len(dist)):
            indice = np.argsort(dist[i])[:T]
            temp = np.concatenate((qf[i][np.newaxis, :], gf[indice]), axis=0)
            qf_new.append(np.mean(temp, axis=0, keepdims=True))

        qf = np.squeeze(np.array(qf_new))
        # feature norm
        q_n = np.linalg.norm(qf, axis=1, keepdims=True)
        qf = qf / q_n

        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        result(dist, self.queryset.ids, self.testset.ids, title='query_expansion')

    def vis(self):
        
        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(self.model, [(torch.unsqueeze(data.query_image, 0), 1)]).numpy()
        gallery_feature = extract_feature(self.model, data.test_loader).numpy()

        # Compute pairwise distance using cdist
        distances = cdist(query_feature, gallery_feature, 'euclidean')
        distances = distances[0]  # as query_feature is just one sample

        # Sort indices in ascending order (smallest distances first)
        indices = np.argsort(distances)

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[indices[i]]
            label = gallery_label[indices[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(str(label))

        fig.savefig("show.png")
        print('result saved to show.png')


if __name__ == '__main__':
    data = Data()
    model = SimpleResNet()
    loss = ResnetLoss()
    main = Main(model, loss, data)

    if opt.mode == 'train':
        for epoch in range(1, 301):
            print('\nepoch', epoch)
            main.train()
            if epoch % 50 == 0:
                os.makedirs('weights', exist_ok=True)
                torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))
            # if epoch % 50 == 0:
            #     print('\nstart evaluate')
            #     main.evaluate(epoch)
            #     os.makedirs('weights', exist_ok=True)
            #     torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate(300)

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()