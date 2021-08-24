# Scraping images from Google Image Source
# See instruction file "install-google-image-download.txt"

#TODO: after downloading images:
# 1. remove non-jpg or non-png images by:
#       find . -name *.webp -type f -delete
#       find . -name *.gif -type f -delete
# 2. removing images if their sizes are lower than 20KB by:
#       find . -size -20k -type f -delete

from google_images_download import google_images_download

# Init
response = google_images_download.googleimagesdownload()

# Get all action names
actions = ""
# read action classes from a file
# ucf101_class_index or hmdb51_class_index
with open("ucf101_class_index_renamed.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        actions = actions + line.strip() + ","

output_dir_ucf = "/Volumes/Kellan/datasets/data_KG_GNN_GCN/ucf101_images_400"
chromedriver_dir = "/Users/Kellan/Desktop/chromedriver"

# creating list of arguments
# 100 images per class
arguments = {"keywords": actions,
             "limit": 200,
             "format": "jpg",
             "chromedriver": chromedriver_dir,
             "output_directory": output_dir_ucf}

# passing the arguments to the function
paths = response.download(arguments)
# printing absolute paths of the downloaded images
print(paths)

