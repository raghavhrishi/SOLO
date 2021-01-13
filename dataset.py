## Author: Lishuo Pan 2020/4/18
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import torchvision 
from matplotlib.patches import Rectangle
import random


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
      self.images_h5 = h5py.File(path[0],mode='r')
      self.image_data = self.images_h5.get('data')
      self.mask_h5 = h5py.File(path[1],mode='r')
      self.mask_data = self.mask_h5.get('data')
      self.bbox_data = np.load(path[3], allow_pickle=True)
      self.label_data = np.load(path[2], allow_pickle=True)
      #self.transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      self.totensor = transforms.ToTensor()

      self.struct_mask = self.grouping_mask_label(self.label_data, self.mask_data)


      
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox

    def grouping_mask_label(self,labels, masks):
        mask_group =[]
        items = 0
        for i in range(len(labels)):        
            mask_group.append(masks[items:items+len(labels[i])])

            items += len(labels[i])
        return np.array(mask_group)
    
    # def grouping_mask_label(self,labels, masks):
    #     final_mask =[]
    #     items = 0
    #     for i in range(len(labels)):        
    #         if len(labels[i]) > 1:
    #             final_mask.append(masks[items:items+len(labels[i])])
    #         else:
    #             final_mask.append(np.asarray([masks[items]]))
    #         items = items + len(labels[i])
    #     return np.array(final_mask)

    def __getitem__(self, index):
        # TODO: __getitem__
        # check flag
        image_data = self.image_data[index]
        label_data = self.label_data[index]
        bbox_data = self.bbox_data[index]
        mask_data = self.mask_data[index]
        mask_list = self.struct_mask[index]
        image_data,mask_list ,bbox_data= self.pre_process_batch(image_data,mask_list,bbox_data)
        assert image_data.shape == (3, 800, 1088)
        print("Initial List shape",bbox_data.shape[0])
        print("Masked List shape",mask_list.shape[0])
        if bbox_data.shape[0] == mask_list.shape[0]:
            print("Success!!")
        assert bbox_data.shape[0] == mask_list.shape[0]

        # for j in range(len(label_data[i])):
        #         print("SCALE",scale)
        #         #bbox = bbox.copy().astype(float)
        #         bbox_data[i][j][0] = np.multiply(bbox_data[i][j][0], x_scale )
        #         bbox_data[i][j][1] = np.multiply(bbox_data[i][j][1], y_scale )
        #         bbox_data[i][j][2] = np.multiply(bbox_data[i][j][2], x_scale)
        #         bbox_data[i][j][3] = np.multiply(bbox_data[i][j][3], y_scale )

        return image_data,label_data,mask_list,bbox_data

    def __len__(self):
        return len(self.image_data)
    
    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        img = img.astype('float64')
        img = self.totensor(img)

        mask = mask.astype('float64')
        mask = self.totensor(mask)

        img = img.permute(1, 2, 0)
        print("Image shape 1",img.shape)
        img = img.permute(0, 2, 1)
        print("Image shape 2",img.shape)
        img = F.interpolate(img, size=800)
        img = img.permute(0, 2, 1)
        img = F.interpolate(img, size=1066)

        
        # img = torchvision.transforms.functional.normalize(img,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        img = torch.nn.functional.pad(img,(11,11))
        #Interpolating the mask
        mask = mask.permute(1, 2, 0)
        mask = mask.permute(0, 2, 1)
        mask = F.interpolate(mask, size=800)
        mask = mask.permute(0, 2, 1)
        mask = F.interpolate(mask, size=1066)
        mask = torch.nn.functional.pad(mask,(11,11))
        # # check flag
        if img.shape == (3, 800, 1088):
          print("Works")
        
        assert img.shape == (3, 800, 1088)

        assert bbox.shape[0] == mask.shape[0]
        x_scale = 800 / 300
        y_scale = 1088 / 400
        print("BBOX SHAPE",bbox.shape)
        # pdb.set_trace()
        bbox[:,0] = np.multiply(bbox[:,0], x_scale)
        bbox[:,1] = np.multiply(bbox[:,1], y_scale)
        bbox[:,2] = np.multiply(bbox[:,2], x_scale)
        bbox[:,3] = np.multiply(bbox[:,3], y_scale)

        return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers      

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        for transed_img, label, transed_mask, transed_bbox in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
        return torch.stack(transed_img_list, dim=0), label_list, transed_mask_list, transed_bbox_list
      

    def loader(self):
      return DataLoader(self.dataset, batch_size=self.batch_size,shuffle=True, collate_fn=self.collect_fn)