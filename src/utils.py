from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np


user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 


def download_image(url):
    respose = request.Request(url,None,headers)
    with request.urlopen(respose) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def load_preprocess_img(img_path):
    if img_path.startswith('https') or img_path.startswith('http'):
        img = download_image(img_path)
    else:
        # Load the image
        img = Image.open(img_path)

    img = img.resize((224,224))

    # Convert to numpy array
    x = np.array(img)
    # Convert x into a batch
    X = np.array([x], dtype='float32')
    # Preprocess Image
    X /= 255.0
    return X