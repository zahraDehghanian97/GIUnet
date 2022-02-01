import torch
from tqdm import tqdm
import torch.optim as optim
from utils.dataset import GraphData
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns

markers = ['o', '*', '+']
sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 10)
colors = ['b', 'r', 'g']


def svm(train_embeddings, train_labels, test_embeddings, test_labels):
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    model.fit(train_embeddings, train_labels)
    predictions = model.predict(test_embeddings)
    true_positive = 0
    for i in range(len(test_labels)):
        if predictions[i] == test_labels[i]:
            true_positive += 1
    accuracy = true_positive / len(test_embeddings)
    return accuracy


def logistic_regression(train_embeddings, train_labels, test_embeddings, test_labels):
    model = LogisticRegression(random_state=0).fit(train_embeddings, train_labels)
    predictions = model.predict(test_embeddings)
    true_positive = 0
    for i in range(len(test_labels)):
        if predictions[i] == test_labels[i]:
            true_positive += 1
    accuracy = true_positive / len(test_embeddings)
    return accuracy



def knn(train_embeddings, train_labels, test_embeddings, test_labels, n):
    model = KNeighborsClassifier(n_neighbors=n).fit(train_embeddings, train_labels)
    predictions = model.predict(test_embeddings)
    true_positive = 0
    for i in range(len(test_labels)):
        if predictions[i] == test_labels[i]:
            true_positive += 1
    accuracy = true_positive / len(test_embeddings)
    return accuracy


def plot_3d(points, labels, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(points.shape[0]):
        ax.scatter(points[i][0], points[i][1], points[i][2], c=colors[labels[i]], marker=markers[labels[i]])
    ax.legend()
    title = 'Embeddings in 3D ' + title
    plt.title(title)
    plt.savefig(title)
    plt.show()
    pass


def plot_2d(points, labels, title):
    df1 = pd.DataFrame(points, columns=['X', 'Y'])
    df2 = pd.DataFrame(labels, columns=['labels'])
    df_total = pd.concat([df1,df2], axis=1)
    title = 'Embeddings in 2D '+ title
    sns.scatterplot(data=df_total, x="X", y="Y", hue="labels", style='labels').set_title(title)
    plt.savefig(title)
    pass



class Trainer:
    def __init__(self, args, net, G_data):
        self.args = args
        self.net = net
        self.feat_dim = G_data.feat_dim
        self.fold_idx = G_data.fold_idx
        self.init(args, G_data.train_gs, G_data.test_gs)
        if torch.cuda.is_available():
            self.net.cuda()

    def init(self, args, train_gs, test_gs):
        print('#train: %d, #test: %d' % (len(train_gs), len(test_gs)))
        train_data = GraphData(train_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        self.train_d = train_data.loader(self.args.batch, True)
        self.test_d = test_data.loader(self.args.batch, False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, amsgrad=True,
            weight_decay=0.0008)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        return gs

    def run_epoch(self, epoch, data, model, optimizer):
        losses, accs, n_samples = [], [], 0
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, hs, ys = batch
            gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
            loss, acc = model(gs, hs, ys)
            losses.append(loss*cur_len)
            accs.append(acc*cur_len)
            n_samples += cur_len
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss, avg_acc = sum(losses) / n_samples, sum(accs) / n_samples
        return avg_loss.item(), avg_acc.item()

    def train(self):
        max_acc = 0.0
        train_str = 'Train epoch %d: loss %.5f acc %.5f'
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f'
        line_str = '%d:\t%.5f\n'
        for e_id in range(self.args.num_epochs):
            self.net.train()
            loss, acc = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer)
            print(train_str % (e_id, loss, acc))

            with torch.no_grad():
                self.net.eval()
                loss, acc = self.run_epoch(e_id, self.test_d, self.net, None)
            if(max_acc < acc):
                max_acc = acc
                train_embeddings = []
                train_labels = []
                test_embeddings = []
                test_labels = []
                for batch in self.train_d:
                    cur_len, gs, hs, ys = batch
                    gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
                    _, _ = self.net(gs, hs, ys)
                    train_embeddings += self.net.embedding
                    train_labels += ys
                    # train_embeddings += net.embedding
                train_labels = list(map(lambda x: x.tolist(), train_labels))
                train_embeddings = list(map(lambda x: x.tolist(), train_embeddings))
                train_embeddings = np.array(train_embeddings)

                for batch in self.test_d:
                    cur_len, gs, hs, ys = batch
                    gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
                    _, _ = self.net(gs, hs, ys)
                    # test_embeddings += net.embedding
                    test_embeddings += self.net.embedding
                    test_labels += ys
                # print(len(test_embeddings))
                # print(test_embeddings[0].detach().numpy().shape)
                # print(net.embedding_best_test)
                # test_embeddings = net.embedding_best_test
                test_labels = list(map(lambda x: x.tolist(), test_labels))
                test_embeddings = list(map(lambda x: x.tolist(), test_embeddings))
                test_embeddings = np.array(test_embeddings)

                # Plotting train embeddings in 2 and 3 dimention
                tsne1 = TSNE(n_components=3, init='random')
                transformed_3d = tsne1.fit_transform(train_embeddings)
                tsne2 = TSNE(n_components=2, init='random')
                transformed_2d = tsne2.fit_transform(train_embeddings)
                plot_2d(transformed_2d, train_labels, 'for train data of IMDB-M-local')
                plot_3d(transformed_3d, train_labels, 'for train data of IMDB-M-local')

                # Plotting test embeddings in 2 and 3 dimention
                tsne1 = TSNE(n_components=3, init='random')
                transformed_3d = tsne1.fit_transform(test_embeddings)
                tsne2 = TSNE(n_components=2, init='random')
                transformed_2d = tsne2.fit_transform(test_embeddings)
                plot_2d(transformed_2d, test_labels, 'for test data of IMDB-M-local')
                plot_3d(transformed_3d, test_labels, 'for test data of IMDB-M-local')

                accuracy = svm(train_embeddings, train_labels, test_embeddings, test_labels)
                print('SVM accuracy', accuracy)

                accuracy = logistic_regression(train_embeddings, train_labels, test_embeddings, test_labels)
                print('LR accuracy', accuracy)

                accuracy = knn(train_embeddings, train_labels, test_embeddings, test_labels, 3)
                print('KNN accuracy', accuracy)

                accuracy = knn(train_embeddings, train_labels, test_embeddings, test_labels, 5)
                print('KNN accuracy', accuracy)
            print(test_str % (e_id, loss, acc, max_acc))

        with open(self.args.acc_file, 'a+') as f:
            f.write(line_str % (self.fold_idx, max_acc))
