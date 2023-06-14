[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/xrP3eqM@C)
# E=mCheese^2^
________________________________________
## ACQUIRING DATA
In src/main call initial_download(). This function creates a new directory with path data/processed. It downloads all the data from boto3.

___________________________________________________
## APPLYING WATERSHED
In src/main call generate_watershed().
Parameters:
bool file_on_prem: if true will pull source images from local disk if false will pull images from aws
string save_local string: default is none which will not save the images will show matplot of the images instead, if given a string will instead save the images in provided directory
bool masks_on_image: if true overlays the mask over the original tif files and saves as tif if false just saves the numpy representation as a txt file
______________________________________________________
## VERIFYING WATERSHED



____________________________________________________
## AUTOENCODER










___________________________________________________
## t-SNE
