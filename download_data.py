"""Script used to download data
"""
import gdown
import os

url = "https://drive.google.com/uc?export=download&confirm=KdYt&id=15CsrVW3HbhXqYv666z7zSIYw-jnJk9Iy"
gdown.download(url, "data.tar.gz", quiet=False)

if os.name == 'posix':
    if os.path.isdir("data"):
        os.system('rm -r data')
    os.system("tar -xvzf data.tar.gz")
    os.system("mv output data")
    os.system("rm data.tar.gz")
