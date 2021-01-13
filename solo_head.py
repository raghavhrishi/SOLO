import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
from scipy.ndimage import measurements

class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        # initial the two branch head modulelist
        self.cate_conv_layer = nn.Sequential(
            nn.Conv2d(256,256,3,stride=1,padding=1,bias=False),
            nn.GroupNorm(num_groups,256),
            nn.ReLU()
        )
        self.cate_head = nn.ModuleList([self.cate_conv_layer for i in range(7)])
        self.cate_out = nn.Sequential(
            nn.Conv2d(256,3,3,padding=1,bias=True),
            nn.Sigmoid()
        )
        
        self.ins_conv1_layer = nn.Sequential(
            nn.Conv2d(258,256,3,stride=1,padding=1,bias=False),
            nn.GroupNorm(num_groups,256),
            nn.ReLU()
        )
        self.ins_head = [self.ins_conv1_layer]
            
        self.ins_conv_layer = nn.Sequential(
            nn.Conv2d(256,256,3,stride=1,padding=1,bias=False),
            nn.GroupNorm(num_groups,256),
            nn.ReLU()
        )
        self.ins_head.extend(nn.ModuleList([self.ins_conv_layer for i in range(6)]))
        self.ins_out_list = [nn.Sequential(nn.Conv2d(256,numGrid**2,1,padding=0,bias=True),
                                           nn.Sigmoid()) for numGrid in self.seg_num_grids]

    # This function initialize weights for head network
    def _init_weights(self):
        ## TODO: initialize the weights
        i = 0
        j = 0
        while i < len(self.cate_head) and j < len(self.ins_head):
            if i < len(self.cate_head):
                nn.init.xavier_uniform_(self.cate_head[i][0].weight)
                i = i+1
            if j < len(self.ins_head):
                nn.init.xavier_uniform_(self.ins_head[j][0].weight)
                j = j+1


    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn#             print('H_FEAT',H_FEAT)
#             print('W_FEAT',W_FEAT)
#             print('FRAC_H',FRAC_H)
#             print('FRAC_W',FRAC_W)
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self, fpn_feat_list, eval=False):
#         for i in range(len(fpn_feat_list)):
# #             print('NEW',new_fpn_list[i].shape)
#             print('OLD',fpn_feat_list[i].shape)
#         print()
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        for i in range(len(new_fpn_list)):
            print('NEW',new_fpn_list[i].shape)
        print()
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
#         print('Quart Shape',quart_shape)
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level,
                                                        new_fpn_list,
                                                        list(range(len(new_fpn_list))),
                                                        eval = eval,
                                                        upsample_shape = quart_shape)
        
        assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):

        new_fpn_list = fpn_feat_list
        new_fpn_list[0] = torch.nn.functional.interpolate(fpn_feat_list[0],scale_factor=0.5,mode='bilinear')
        new_fpn_list[-1] = torch.nn.functional.interpolate(fpn_feat_list[-1],size=(25,34),mode='bilinear')
        return new_fpn_list

    # This function forward a single level of fpn_feat map through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        print(idx)
        print('fpn_feat',fpn_feat.shape)
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.seg_num_grids[idx]  # current level grid
        print('category before pass: ',cate_pred.shape)
        for i,layer in enumerate(self.cate_head):
            cate_pred = layer(cate_pred)
            
        cate_pred = self.cate_out(cate_pred)
        print('category tensor after pass: ',cate_pred.shape)
        
        bz = fpn_feat.shape[0]
        H_FEAT = fpn_feat.shape[2]
        W_FEAT = fpn_feat.shape[3]
        yv, xv = torch.meshgrid([torch.linspace(-1,1,H_FEAT), torch.linspace(-1,1,W_FEAT)])
        yv = yv.view(1,1,H_FEAT,W_FEAT)
        yv = torch.cat([yv]*bz)
        xv = xv.view(1,1,H_FEAT,W_FEAT)
        xv = torch.cat([xv]*bz)
        print('xv: ',xv.shape)
        print('yv: ',yv.shape)

        ins_pred = torch.cat([ins_pred,xv,yv],1)
#         torch.cat([ins_pred[1],xv,yv],0)
        print('ins_pred tensor after concat: ', ins_pred.shape)
        for i,layer in enumerate(self.ins_head):
            ins_pred = layer(ins_pred)
        ins_pred = self.ins_out_list[idx](ins_pred)
        print('ins_pred after forward pass: ', ins_pred.shape)
        
        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO resize ins_pred            
            cate_pred = torch.nn.functional.interpolate(cate_pred,size=num_grid,mode='bilinear')
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)
            print('eval=false modification for cat: ',cate_pred.shape)
            
            ORI_H = 800
            ORI_W = 1088
            FRAC_H = (ORI_H/4)/H_FEAT
            FRAC_W= (ORI_W/4)/W_FEAT
            print('eval=false modification for ins (BEFORE): ', ins_pred.shape)
            ins_pred = torch.nn.functional.interpolate(ins_pred,size=(upsample_shape[0],upsample_shape[1]),mode='bilinear')
            print('eval=false modification for ins (AFTER): ', ins_pred.shape)

        # check flag
        if eval == False:
            cate_pred = torch.nn.functional.interpolate(cate_pred,size=num_grid,mode='bilinear')
            print('eval=false modification for cat: ',cate_pred.shape)
            
            print('eval=false modification for ins (BEFORE): ', ins_pred.shape)
            ins_pred = torch.nn.functional.interpolate(ins_pred,size=(2*H_FEAT,2*W_FEAT),mode='bilinear')
            print('eval=false modification for ins (AFTER): ', ins_pred.shape)
            
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        print()
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        pass



    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        pass

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        pass

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,ins_pred_list,bbox_list,label_list,mask_list):
        
        #final_prediction = [] 
        # for i in range(len(ins_pred_list)):
        #     final_prediction.append((ins_pred_list[1],ins_pred_list[2],ins_pred_list[3]))
        
        feat_map_list = [[200,272],[200,272],[100,136],[50,68],[50,68]]
        ins_gts_list,ins_ind_gts_list,cate_gts_list = self.MultiApply(self.target_single_img,bbox_list,label_list,mask_list, featmap_sizes= feat_map_list)
        
        # # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # # remember, you want to construct target of the same resolution as prediction output in training

        # # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image
        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []
        for i in range(len(gt_bboxes_raw)):
            
            x1,y1,x2,y2 = gt_bboxes_raw[i]
            label_val = gt_labels_raw[i]
            mask_val = gt_masks_raw[i]
            # mask_val = mask_val.double().numpy()
            instance_scale = ((y2-y1)*(x2-x1))**0.5
            #Calculating center of mass and appending to the list

            for index in range(len(featmap_sizes)):

                #Extracting the h and w value and doubling them 
                double_h_val = featmap_sizes[index][0]
                double_w_val = featmap_sizes[index][1]
                #Taking the s value 
                s = self.seg_num_grids[index]
                #Min and Max range based on FPN layers 
                minimum_scale = self.scale_ranges[index][0]
                maximum_scale = self.scale_ranges[index][1]
                #Initializing the output
                cate_label = torch.zeros(s,s)
                ins_label = torch.zeros(s*s,double_h_val,double_w_val)
                ins_in_label = torch.zeros(s*s)

                #Check the ranges for the instance scale
                if instance_scale > minimum_scale and instance_scale < maximum_scale:
                    center_height,center_width = measurements.center_of_mass(mask_val.double().numpy())
                    #Fetching the height and width of the mask 
                    ori_h =  mask_val.shape[0]
                    ori_w =  mask_val.shape[1]
                    #Calculating the epsilon height and the width(multiplying epsilon with ori_h and ori_w)
                    epsilon_height = self.epsilon*ori_h/2
                    epsilon_width = self.epsilon*ori_w/2
                    #Calculating the image center 
                    center_grid_height = int(center_height/(epsilon_height*s))
                    center_grid_width = int(center_width/(epsilon_width*s))
                    #Coordinates calculation for the grid 
                    top = max(0,int((center_height-epsilon_height)/ori_h*s))
                    bottom = max(s-1, int((center_height+ epsilon_height)/(ori_h*s)))
                    left = max(0,int((center_width-ori_w)/(ori_w*s)))
                    right = max(s-1,int((center_width+ori_w)/(ori_w*s)))
                    #Constraining it within 3x3 grid 
                    top_val = max(top,center_grid_height - 1)
                    bottom_val = max(bottom,center_grid_height + 1)
                    left_val = max(left,center_grid_width - 1)
                    right_val = max(right,center_grid_width + 1)
                    #Unsqueezing the mask twice to get the right dimension
                    mask_val = torch.unsqueeze(mask_val,0)
                    mask_val = torch.unsqueeze(mask_val,0)
                    scaling_gt_mask = F.interpolate(mask_val,size=(double_h_val,double_w_val),mode='bilinear')
                    mask_val = torch.squeeze(mask_val,0)
                    mask_val = torch.squeeze(mask_val,0)
                    for i in range(top_val,bottom_val+1):
                        for id in range(left_val,right_val+1):
                            combined_ij = i*s + id
                            ins_label[combined_ij]=scaling_gt_mask
                            ins_in_label[combined_ij]=1
                            cate_label[i][id] = label_val

                    
                ins_label_list.append(ins_label)
                ins_ind_label_list.append(ins_in_label)
                cate_label_list.append(cate_label)
                print(i)
                print('ins_label.shape: ',ins_label.shape)
                print('ins_in_label.shape: ',ins_in_label.shape)
                print('cate_label.shape: ',cate_label.shape)
                print()

        # check flag
        # print('ins_label_list.shape: ',ins_label_list[1].shape)
        # print('ins_ind_label_list[1].shape: ',ins_ind_label_list[1].shape)
        # print('cate_label_list[1].shape: ',cate_label_list[1].shape)


        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)

        return ins_label_list, ins_ind_label_list, cate_label_list


    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        ## TODO: finish PostProcess
        pass


    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        ## TODO: PostProcess on single image.
        pass

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        pass

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,ins_gts_list,ins_ind_gts_list,cate_gts_list,color_list,img):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.

        #  For the first image, can be any image
        #Getting the number of levels of the feature map
        # pdb.set_trace()
        for val in range(len(ins_gts_list)):
            ins_val = ins_gts_list[val]
            ins_ind_val = ins_ind_gts_list[val]
            cate_val = cate_gts_list[val]
            img_val = img[val]
            factor = int(len(ins_gts_list[val])/5)
            for idx, grid in enumerate(self.seg_num_grids):
        #         img_val = transforms.functional.normalize(img_val, (-0.485,-0.456,-0.406),(1/0.229,1/0.224,1/0.225))
                img_plot = img_val.data.numpy().transpose((1,2,0)).astype('int')
                plt.imshow(img_plot)
                for i in range(grid):
                    for j in range(grid):
                        for k in range(int(len(ins_gts_list[val])/5)):
                            if ins_ind_gts_list[val][idx+k*5][i*grid+j] == 1:
                                mask_val = ins_gts_list[val][idx+k*5][i*grid+j]
                                category = cate_gts_list[val][idx+k*5][i][j].double()
                                color = color_list[int(category)-1]
                                mask_val = torch.unsqueeze(mask_val,0)
                                mask_val = torch.unsqueeze(mask_val,0)
                                mask_val = nn.functional.interpolate(mask_val,size=(800,1088),mode='bilinear')
                                mask_val = mask_val.squeeze(0)
                                mask_val = mask_val.squeeze(0).numpy()
                                mask_re = np.reshape(mask_val,(800,1088,1))
                                plt_mask = np.ma.masked_where(mask_re==0,mask_re)
                                plt_mask = np.squeeze(plt_mask)
    #                             plt_mask_list.append(plt_mask)
                                plt.imshow(plt_mask,cmap=color,alpha=0.5)
                # plt.savefig("fig-%d-fpn-%d" % (val,idx))
                plt.show()
        

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        pass

from backbone import *
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img.float())
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)
