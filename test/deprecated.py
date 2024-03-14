import numpy as np

def check_trail(img,axis=0):
    sigma = np.std(img)
    # sigma = median_abs_deviation(img,axis=None)
    sorted = np.sort(img,axis=axis)
    if axis == 0:
        n1 = np.sum(sorted[-5,:]>0)
        n2 = np.sum(sorted[-5,:]>5*sigma)
    if axis == 1:
        n1 = np.sum(sorted[:,-5]>0)
        n2 = np.sum(sorted[:,-5]>5*sigma)

    print("%i/%i=%.3f"%(n2,n1,n2/n1))