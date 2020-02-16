"""Script used to download data
"""
import gdown
import os

url = "https://drive.google.com/uc?id=1NLWLUMM0OzwLb40GA1w7vnP02pGc0urM"
gdown.download(url, "data.tar.gz", quiet=False)

if os.name == 'posix':
    if os.path.isdir("data"):
        os.system('rm -r data')
    os.system("tar -xvzf data.tar.gz")
    os.system("mv output data")
    os.system("rm data.tar.gz")
