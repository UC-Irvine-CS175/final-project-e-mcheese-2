[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/xrP3eqM@C)
# E=mCheese<sup>2</sup>
________________________________________
## ACQUIRING DATA
In src/main.py call initial_download().  
This function creates a new directory with path data/processed. It downloads all the data from AWS using boto3.  
  
**NOTE: ONLY CALL THIS FUNCTION ONCE!**

___________________________________________________
## APPLYING WATERSHED
In src/main.py call generate_watershed().  
This function generates all of the watershed masks for a given subset of images.  

Parameters:  
__string csv_dir__: The directory that contains the target csv file. This has to include the full path to the directory.  
__string csv_file__: The name of the csv file contained in csv_dir with the target images.  
__bool file_on_prem__: If true will pull source images from local disk. If false will pull images from AWS.  
__string save_local__: Default is none, which will not save the images but instead will show temporary matplot images. If given a string, it should be the directory where the images will be saved to.  
______________________________________________________
## VERIFYING WATERSHED
To produce plots of foci counts and estimated Poisson distributions, run the main() function of src/poisson_distributions.py.
In order to run this main, the following steps must be completed:

Use prepare_subset_data() to create CSV files for each subset.
Parameters:  
__string meta_csv_path__: The path to the meta csv file containing the entire dataset.  
__string subset_csv_dir__: The path to the directory where the user wants to save the subset csv files.  

Use pickle_masks() on each subset CSV to create an array of masks for each image in the CSV and save that array as a pickle file locally.
Parameters:  
__string local_csv_path__: The path to the subset csv file.  
__string local_data_dir__: The path to the directory where raw images are stored.  
__string local_masks_path__: The path to where the user wants the mask pickle to be saved.  
__Watershed watershed__: The path to the directory where the user wants to save the subset csv files.  
__torchvision.transform transform__: Transformation to apply to images before masking. Default None.  

In the poisson_distributions.py main(), change the directory passed to visualize_subset() to the local mask directory (whatever was passed as subset_csv_dir to prepare_subset_data()).


____________________________________________________
## TRAINING THE AUTOENCODER
In src/main.py or src/models/unsupervised/autoencoder_v1.py call train_autoencoder().  
This function trains the unsupervised autoencoder model.  
  
Parameters:  
__string csv_dir__: The directory that contains the target csv file. This has to include the full path to the directory.  
__string train_file__: The name of the csv file contained in csv_dir with the images to train the model with  
__string val_file__: The name of the csv file contained in csv_dir with the images to validate the model with  
__int num_workers__: Specifies how many workers will be used for the dataloaders  
  
**NOTE: All files to be used for training/validation must be stored locally in csv_dir**

____________________________________________________
## USING THE AUTOENCODER
In src/main.py call generate_autoencoder_output().  
This function loads a pretrained autoencoder model, and generates its output images for a given subset.  
  
Parameters:  
__model_weights__: Specifies the path to the trained model weights.  
__string csv_dir__: The directory that contains the target csv file. This has to include the full path to the directory.  
__string csv_file__: The name of the csv file contained in csv_dir with the target images.  
__string save_local__: Default is none, which will not save the images but instead will show temporary matplot images. If given a string, it should be the directory where the images will be saved to.  

**NOTE: There are pretrained model weights in the directory named Model_Weights**




___________________________________________________
## t-SNE Analysis
In src/main.py call generate_tsne_plots().  
This function saves the t-SNE plots for the images in a given directory and subset.  
  
Parameters:  
__string csv_dir__: The directory that contains the target csv file. This has to include the full path to the directory.  
__string csv_file__: The name of the csv file contained in csv_dir with the target images.  
__string file_name__: This specifies the titles and file names of the t-SNE plots

**NOTE: Make sure the files you want to evaluate are in csv_dir, as only the files in this directory will be found.**  


____________________________________________________
## Comparing All Model Images
In src/main.py call plot_all_images().  
This function shows a side by side comparison of the raw images, watershed masks, and autoencoder output.  
  
Parameters:  
__string process_dir__: The directory that contains the target csv file, along with all the raw images.   
__string watershed_dir__: The directory that contains the watershed output images.  
__string autoencoder_dir__: The directory that contains the autoencoder output images.  
__string save_local__: Default is none, which will not save the images but instead will show temporary matplot images. If given a string, it should be the directory where the images will be saved to. 

**NOTE: All of the directory parameters must include the full path to each respective directory.**  


____________________________________________________
## Tips on Saving Output Images
**Make sure to save all raw, Watershed, and Autoencoder Outputs into separate directories**  





















