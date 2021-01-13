
from dataset import *

## Visualize debugging
if __name__ == '__main__':
    #Loading Data 
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    dataset = BuildDataset(paths)
    #image_data,label_data,mask_data,bbox_data = dataset.__getitem__(9)
    full_size = len(dataset)
    # #Size Initialization 
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    print("Train Size",train_size)
    print("Test Size",test_size)
    torch.random.manual_seed(1)
    # #Splitting Data 
    batch_size = 10
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_build_loader = BuildDataLoader(train_dataset, batch_size= 10, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size= 10, shuffle=True, num_workers=0)
    test_loader = test_build_loader.loader()
    mask_color_list = ["jet", "cool", "Spectral", "Spectral", "ocean"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Train loader",type(train_loader))
    for iter, data in enumerate(train_loader, 0):
        print("Training Starts!")
        img, label, mask, bbox = [data[i] for i in range(len(data))]
        print("Image shape",img.shape)
        
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        if img.shape == (batch_size, 3, 800, 1088):
            print("Image shape")

        assert len(mask) == batch_size
        if len(mask) == batch_size:
            print("Mask")
    

        # plot the origin img
        x_scale = 800 / 300
        y_scale = 1088 / 400
        for i in range(batch_size):
            print("Mask Shape",mask[i].shape)
            ## TODO: plot images with annotations
            #Plotting still to be done

            image_plot = img[i]
            print("Image shape",image_plot.shape)
            image_plot = np.transpose(image_plot.numpy(), (1,2,0)).astype('long')
            plt.figure(figsize = (7,10))

            print(bbox[i].shape)
            plt.imshow(image_plot)
            print("MASKED SHAPE",mask[i].shape)
            print("Bounding Box SHAPE",bbox[i].shape)
            channel_dim = mask[i].shape

            if channel_dim[0] > 1 and channel_dim[0] < 3:
                mask1 = mask[i][0]
                mask1 = np.reshape(mask1,(800,1088))
                masked1 = np.ma.masked_where(mask1 == 0, mask1)
                colors = mask_color_list[label[i][0].item()]
                plt.imshow(masked1, cmap=colors, alpha=0.5) # interpolation='none'
                mask2 = mask[i][1]
                mask2 = np.reshape(mask2,(800,1088))
                masked2 = np.ma.masked_where(mask2 == 0, mask2)
                colors = mask_color_list[label[i][1].item()]
                plt.imshow(masked2, cmap=colors, alpha=0.5) # interpolation='none'

            elif channel_dim[0] == 3:
                mask1 = mask[i][0]
                mask1 = np.reshape(mask1,(800,1088))
                masked1 = np.ma.masked_where(mask1 == 0, mask1)
                colors = mask_color_list[label[i][0].item()]
                plt.imshow(masked1, cmap=colors, alpha=0.5) # interpolation='none'
                mask2 = mask[i][1]
                mask2 = np.reshape(mask2,(800,1088))
                masked2 = np.ma.masked_where(mask2 == 0, mask2)
                colors = mask_color_list[label[i][1].item()]
                plt.imshow(masked2, cmap=colors, alpha=0.5) # interpolation='none'
                mask3 = mask[i][2]
                mask3 = np.reshape(mask3,(800,1088))
                masked3 = np.ma.masked_where(mask3 == 0, mask3)
                colors = mask_color_list[label[i][2].item()]
                plt.imshow(masked3, cmap=colors, alpha=0.5) # interpolation='none'
                

            else:
                mask[i] = np.reshape(mask[i],(800,1088))
                masked = np.ma.masked_where(mask[i] == 0, mask[i])
                colors = mask_color_list[label[i].item()]
                plt.imshow(masked, cmap=colors, alpha=0.5) # interpolation='none'
            
            ax = plt.gca()


            for j in range(len(label[i])):
                #bbox = bbox.copy().astype(float)
                bbox[i][j][0] = np.multiply(bbox[i][j][0], x_scale )
                bbox[i][j][1] = np.multiply(bbox[i][j][1], y_scale )
                bbox[i][j][2] = np.multiply(bbox[i][j][2], x_scale)
                bbox[i][j][3] = np.multiply(bbox[i][j][3], y_scale )
                rect2 = patches.Rectangle((bbox[i][j][0],bbox[i][j][1]),bbox[i][j][2] - bbox[i][j][0],bbox[i][j][3] - bbox[i][j][1],linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rect2)
                   
            # plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
            plt.show()  

        if iter == 10:
            break