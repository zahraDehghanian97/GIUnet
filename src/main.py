import argparse
import random
import time
import torch
import numpy as np
from network import GNet
from trainer import Trainer
from utils.data_loader import FileLoader


def get_args():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=1, help='seed')
    parser.add_argument('-data', default='MUTAG', help='data folder name')
    parser.add_argument('-fold', type=int, default=1, help='fold (1..10)')
    parser.add_argument('-num_epochs', type=int, default=100, help='epochs')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-deg_as_tag', type=int, default=0, help='1 or degree')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
    parser.add_argument('-act_n', type=str, default='ELU', help='network act')
    parser.add_argument('-act_c', type=str, default='ELU', help='output act')
    parser.add_argument('-ks', nargs='+', type=float, default=[0.9, 0.8, 0.7])
    parser.add_argument('-acc_file', type=str, default='re', help='acc file')
    args, _ = parser.parse_known_args()
    return args


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return


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



from torchviz import make_dot
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


def app_run(args, G_data, fold_idx):
    G_data.use_fold_data(fold_idx)
    net = GNet(G_data.feat_dim, G_data.num_class, args)
    # t_gs = G_data.train_gs
    # data = GraphData(t_gs, G_data.feat_dim)
    # train_d = data.loader(8, True)
    # yhat = net(train_d)
    # make_dot(yhat, params=dict(list(net.named_parameters()))).render("torchviz", format="png")
    trainer = Trainer(args, net, G_data)
    trainer.train()
    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []
    for batch in trainer.train_d:
        cur_len, gs, hs, ys = batch
        gs, hs, ys = map(trainer.to_cuda, [gs, hs, ys])
        _, _ = net(gs, hs, ys)
        train_embeddings += net.embedding
        train_labels += ys
        # train_embeddings += net.embedding
    train_labels = list(map(lambda x: x.tolist(), train_labels))
    train_embeddings = list(map(lambda x: x.tolist(), train_embeddings))
    train_embeddings = np.array(train_embeddings)

    for batch in trainer.test_d:
        cur_len, gs, hs, ys = batch
        gs, hs, ys = map(trainer.to_cuda, [gs, hs, ys])
        _, _ = net(gs, hs, ys)
        #test_embeddings += net.embedding
        test_embeddings += net.embedding
        test_labels += ys
    #print(len(test_embeddings))
    #print(test_embeddings[0].detach().numpy().shape)
    #print(net.embedding_best_test)
    #test_embeddings = net.embedding_best_test
    test_labels = list(map(lambda x: x.tolist(), test_labels))
    test_embeddings = list(map(lambda x: x.tolist(), test_embeddings))
    test_embeddings = np.array(test_embeddings)

    #Plotting train embeddings in 2 and 3 dimention
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

    accuracy = svm(train_embeddings, train_labels,test_embeddings, test_labels)
    print('SVM accuracy', accuracy)

    accuracy = logistic_regression(train_embeddings, train_labels, test_embeddings, test_labels)
    print('LR accuracy', accuracy)

    accuracy = knn(train_embeddings, train_labels, test_embeddings, test_labels, 3)
    print('KNN accuracy', accuracy)

    accuracy = knn(train_embeddings, train_labels, test_embeddings, test_labels, 5)
    print('KNN accuracy', accuracy)


def main():
    args = get_args()
    print(args)
    set_random(args.seed)
    start = time.time()
    G_data = FileLoader(args).load_data()
    print('load data using ------>', time.time() - start)
    if args.fold == 0:
        for fold_idx in range(10):
            print('start training ------> fold', fold_idx + 1)
            app_run(args, G_data, fold_idx)
    else:
        print('start training ------> fold', args.fold)
        app_run(args, G_data, args.fold - 1)


if __name__ == "__main__":
    main()
