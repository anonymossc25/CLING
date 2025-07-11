import dgl
from dgl.data import RedditDataset
dataset = RedditDataset()
g = dataset[0]
num_class = dataset.num_classes

# get node feature
feat = g.ndata['feat']
print(feat)

# get data split
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']

# get labels
label = g.ndata['label']
