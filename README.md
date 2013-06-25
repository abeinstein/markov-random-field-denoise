Homework 6
Andrew Beinstein

hw6.py: Main script
image_processing.py: Contains shape-drawing and other miscealleous
image processing algorithms.
hw6.pdf: My write-up

The main hw6.py file requires two command line arguments:
> python hw6.py <image_filepath> <mu value>
where image_filepath is a path to an ASCII binary image file, and
mu is a floating point number between 0 and 1. My script will then 
introduce noise into the picture with probability mu, and run the 
Loopy BP algorithm to denoise the picture. The script then displays the 
picture using Numpy and Matplotlib, and it outputs the fraction error:
# pixels changed from original / total # pixels

To reconstruct some of the plots from my write-up, scroll to the bottom
of hw6.py, where I have indicated which lines you can uncomment. 

Enjoy!
Andrew
