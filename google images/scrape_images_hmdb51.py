# Scraping images from Google Image Source
# See instruction file "install-google-image-download.txt"

# Probably do:
# scrape over 100 images for each class since the issue from this tool

from google_images_download import google_images_download

# Init
response = google_images_download.googleimagesdownload()

# Get all action names
actions = ""
# read action classes from a file
# ucf101_class_index or hmdb51_class_index
with open("hmdb51_class_index.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        actions = actions + line.strip() + ","

output_dir_hmdb = "/Volumes/Kellan/datasets/data_KG_GNN_GCN/hmdb51_images_400"
chromedriver_dir = "/Users/Kellan/Desktop/chromedriver"

# creating list of arguments
# 100 images per class
arguments = {"keywords": actions,
             "limit": 400,
             "language": "English",
             "format": "jpg",
             "chromedriver": chromedriver_dir,
             "output_directory": output_dir_hmdb}

# passing the arguments to the function
paths = response.download(arguments)
# printing absolute paths of the downloaded images
print(paths)

