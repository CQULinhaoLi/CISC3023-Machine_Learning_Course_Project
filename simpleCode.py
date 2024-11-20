import matplotlib.pyplot as plt
import csv
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error

def readImageData(rootpath):
    '''Reads data 
    Arguments: path to the image, for example './Training'
    Returns:   list of images, list of corresponding outputs'''
    images = [] # images
    output_1 = [] # corresponding x index
    # there are other outputs like y index, x_width, y_width
    
    prefix = rootpath + '/' 
    gtFile = open(prefix + 'myData'+ '.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader)
    # loop over all images in current annotations file
    for row in gtReader:
        img=Image.open(prefix + row[0])  # the 1th column is the filename
        # preprocesing image, here we resize the image into a smaller one
        img=img.resize((32,32), Image.BICUBIC)
        img=np.array(img)
        images.append(img)
        output_1.append(row[1]) # the 8th column is the label
    
    gtFile.close()
    return images, output_1


trainImages, trainOutputs = readImageData('Training')
# print number of historical images
print('number of historical data=', len(trainOutputs))
# show one sample image
plt.imshow(trainImages[4])
plt.show()

# design the input and output for model
X = []
Y = []
for i in range(0, len(trainOutputs)):
    # input X just the flattern image, you can design other features to represent a image
    X.append(trainImages[i].flatten())
    Y.append(int(trainOutputs[i]))

X = np.array(X)
Y = np.array(Y)

#train a Randomforest
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X,Y)
Ypred = reg.predict(X)

#check the accuracy
MSE = mean_squared_error(Y,Ypred)
print('Training MSE=', MSE)

# save model
import pickle
pickle.dump(reg, open('model.sav','wb'))



