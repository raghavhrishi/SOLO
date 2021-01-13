# SOLO - Segmenting Objects by Location 
Original Paper: https://arxiv.org/pdf/1912.04488.pdf
### Instructions to run the code: 

1. Dataset Plotting:  Run the dataset_plot.py file for visualizing the plot for the dataset. This gives specific colours for the three classes (vehicles,animals and people.
In order to run the plot function and visualize the image properly, the image has been denormalized. The masks and bounding boxes are plotted on it. Run it as follows:

        python3 dataset_plot.py

2.  Running `solo_head.py` : In order to run this file, execute 

        python3 solo_head.py
        
    This would start executing the `solo_head.py` python file and would plot the images at the end. 

Note: Please change the paths of the data files
