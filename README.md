# SOFE4620-mproj-3
Mini Project 3 for Machine Learning and Data Mining

To use run pip install -r requirements.txt to get all the required python libraries, then run main.py  

Any variables can be configured in main.py, images are read in the form of label*.bmp, where label is 
the digit represented, * is any characters, and .bmp is the final file type (eg. 0_TestSet1.bmp)  

The testing set used after training the system is similar to the trained values but with 
subtle differences (ie some black pixels may be white pixels). If you add any more values to the training set please
ensure their image dimensions are 5 x 9 pixels and greyscale. If the images are not greyscale it is okay as the data
handler file will make them greyscale.  