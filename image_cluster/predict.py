
from sklearn.cluster import KMeans
import h5py

NUM_CLUSTERS=10

root='./intervel/days_3/'
h5f=h5py.File(root+'feature.h5','r')
feature_data=h5f['feature']

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(feature_data)
predict=kmeans.predict(feature_data)

h5f=h5py.File(root+'predict.h5','w')
h5f.create_dataset('predict',data=predict)
h5f.close()
