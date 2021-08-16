from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

arguments = {"keywords": "Apply Eye Makeup,Apply Lipstick,Archery",
             "limit": 10,
             "print_urls": True}   # creating list of arguments

# passing the arguments to the function
paths = response.download(arguments)
# printing absolute paths of the downloaded images
print(paths)
