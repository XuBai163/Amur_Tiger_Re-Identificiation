from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.random_erasing import RandomErasing
from utils.random_sampler import RandomSampler
from torchvision.transforms import InterpolationMode
from resnet50_mgn_pose.opt_pose import opt
import os
import re
import csv
import json

class Data():
    def __init__(self):
        train_transform = transforms.Compose([
            transforms.Resize((128, 288), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((128, 288), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = CVWC(train_transform, 'train', opt.data_path)
        self.testset = CVWC(test_transform, 'test', opt.data_path)
        self.queryset = CVWC(test_transform, 'query', opt.data_path)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=2,
                                                  pin_memory=True,drop_last = True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=2, pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=2,
                                                  pin_memory=True)

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))


class CVWC(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/atrw_reid_train/train_new'
            self.img2id_dic = self.img2id('files/reid_list_train.csv')
        else:
            self.data_path += '/atrw_reid_test/test_new'
            self.img2id_dic = self.img2id('dataset/atrw_anno_reid_test/reid_list_test.csv')
        
        self.imgs = [path for path in self.list_pictures(self.data_path) if os.path.basename(path) in self.img2id_dic.keys()]

        # self.img2id_dic = self.img2id('files/reid_list_train.csv')
        # # self.imgs = [path for path in self.list_pictures(self.data_path)]
        # self.imgs = [path for path in self.list_pictures(self.data_path) if os.path.basename(path) in self.img2id_dic.keys()]

        self.kpt = json.load(open('files/new_keypoints_train.json'))
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):      
        path = self.imgs[index]

        target = self._id2label[self.id(path)]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

    def img2id(self, csv_path):
        with open(csv_path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
            csv_reader = csv.DictReader(f, fieldnames=fieldnames)
            d = {}
            for row in csv_reader:
                # d[row['img_id']] = int(row['id'])

                if 'img_id' in row and 'id' in row:
                    d[row['img_id']] = int(row['id'])
                else:
                    d[row['img_id']] = int(row['img_id'][-8:-4])

        return d

    def id(self, path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        imgid = os.path.basename(path)
        return self.img2id_dic[imgid] 
        # target = self.img2id_dic[imgid] if imgid in self.img2id_dic.keys() else int(str(imgid[-8:-4]))
        # return int(target)

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))