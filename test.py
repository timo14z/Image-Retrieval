import scipy.spatial.distance
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import shelve
from os import listdir
from os.path import join

DATASET_PATH = "I:\\Projects\\Python\\machine learning course\\images\\shapes"
RELOAD_DATASET = True
CACHE_PATH = "I:\\Projects\\Python\\machine learning course"


model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
# model = VGG16(weights='imagenet', include_top=False, pooling='avg')


# process all images in data set
def get_dataset(re=RELOAD_DATASET):
    cache = shelve.open(CACHE_PATH + "dataset")
    if not re:
        return cache["dataset"]
    data = []
    # get all the images in the folder
    for f in listdir(DATASET_PATH):
        img = extract_features(join(DATASET_PATH, f))
        #print(img)
        data.append((img, join(DATASET_PATH, f)))
    cache["dataset"] = data
    cache.close()
    return data


# extracting features (returns features vector)
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    ret = image.img_to_array(img)
    ret = np.expand_dims(ret, axis=0)
    ret = preprocess_input(ret)
    ret = model.predict(ret)
    ret = np.array(ret).flatten()
    return ret


# Calculate Cosine Similarity
def cosine_dis(x, y):
    return scipy.spatial.distance.cosine(x, y)


# Calculate chi squared Distance
def chi2_distance(x, y, eps=1e-10):
    d = 0.5 * np.sum(((x - y) ** 2) / (x + y + eps))
    return d


def get_results(img, data=get_dataset()):
    img = extract_features(img)
    # init results dict to store each images chi dist from query
    results = []
    # loop over the rows in features
    for cur, path in enumerate(data):
        # calc dist between image i and query image
        #print(data[cur][0])
        dis = chi2_distance(data[cur][0], img)
        # store dist in results dict with image path as key
        results.append((dis, data[cur][1]))
    return results




tstpath = DATASET_PATH + '\\t1.jpg'
tmp = get_results(tstpath)
tmp = sorted(tmp)
for i in range(1, 5):
    img = mpimg.imread(tmp[i][1])
    imgplot = plt.imshow(img)
    plt.title(str(i))
    plt.show()