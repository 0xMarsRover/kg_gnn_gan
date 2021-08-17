# Scraping images from Google Image Source
# See instruction file "install-google-image-download.txt"

# Maybe do:
# scrape over 100 images for each class since the issue from this tool

from google_images_download import google_images_download

# Init
response = google_images_download.googleimagesdownload()

# Get all action names
actions = ""
with open("ucf101_class_index.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        actions = actions + line.strip() + ","
# print(actions)

# creating list of arguments
arguments = {"keywords": actions,
             "limit": 100,
             "print_urls": True}

# passing the arguments to the function
paths = response.download(arguments)
# printing absolute paths of the downloaded images
print(paths)

