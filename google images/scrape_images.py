# Scraping images from Google Image Source
# See instruction file "install-google-image-download.txt"

# Will do:
# reading all action classes (or object names) from a file
# and scraping images for them in one run

from google_images_download import google_images_download

# Init
response = google_images_download.googleimagesdownload()

# creating list of arguments
actions = "Apply Eye Makeup,Apply Lipstick,Archery"
arguments = {"keywords": actions,
             "limit": 10,
             "print_urls": True}

# passing the arguments to the function
paths = response.download(arguments)
# printing absolute paths of the downloaded images
print(paths)
