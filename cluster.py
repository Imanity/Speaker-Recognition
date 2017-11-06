import numpy as np
from feature_extract import getKMeansVec, getKMeansVec2

tang_test = np.load("np/tang1.npy")
tang_predict = np.load("np/tang2.npy")
wei_test = np.load("np/wei1.npy")
wei_predict = np.load("np/wei2.npy")
luo_test = np.load("np/luo1.npy")
luo_predict = np.load("np/luo2.npy")


print(tang_test.shape)
print(tang_predict.shape)
print(getKMeansVec(tang_test))
print("-----------" + "tang_predict" + "------------")
print(getKMeansVec2(tang_predict,3))
print("-----------" + "wei_predict" + "------------")
print(getKMeansVec2(wei_predict,3))
print("-----------" + "luo_predict" + "------------")
print(getKMeansVec2(luo_predict,3))

print("-----------" + "tang_test" + "------------")
print(getKMeansVec2(tang_test,3))
print("-----------" + "wei_test" + "------------")
print(getKMeansVec2(wei_test,3))
print("-----------" + "luo_test" + "------------")
print(getKMeansVec2(luo_test,3))


print("-----------" + "tang3" + "------------")
print(getKMeansVec2(np.load("np/tang3.npy"),3))