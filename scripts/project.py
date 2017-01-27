#-*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import squareform  # coloca no formato de matriz
from functools import lru_cache
from matplotlib import cm
from sklearn.metrics import classification_report, confusion_matrix

import pdb

import os  # funções utilitárias do sistema operacional

import pandas as pd  # manipulação de tabelas, usado para ler formato csv

# Visualization
import matplotlib.pyplot as plt

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False


@lru_cache(maxsize=None)
def dist_euclidean(x, y):
    return np.linalg.norm(x - y)


def dtw(ts_a, ts_b, max_warping_window=100, fdist=dist_euclidean):
    """ Retorna a distância entre duas 'timeseries' 2D.

    ---------
    ts_a, ts_b : ndarrays [n_samples, n_timepoints]
        Arrays contendo séries com n_samples amostras para
        serem comparados pelo DTW

    max_warping_window : tamanho máximo da janela utilizado
    para minimizar o custo de deformação (warping cost)

    d : Métrica de distância, default=euclidean

    Returns
    -------
    A similaridade entre A e B definida pelo algoritmo DTW
    """

    # Cria a matriz de custos, e inicializa cada posição com o valor máximo (custo maximo)

    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxsize * np.ones((M, N))

    # Inicializa a primeira linha e coluna
    cost[0, 0] = fdist(ts_a[0], ts_b[0])
    cost[1:, 0] = [cost[i-1, 0] + fdist(ts_a[i], ts_b[0]) for i in range(1, M)]

    cost[0, 1:] = [cost[0, j-1] + fdist(ts_a[0], ts_b[j]) for j in range(1, N)]

    # Popula o resto da matriz utilizando a janela
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window),
                        min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + fdist(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


def most_common(coll):
    lst = list(coll)
    return max(lst, key=lst.count)


def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res), mns, sstd


class KNN(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN

    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """

    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1, nlabels=8):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
        self.nlabels = nlabels

    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = x
        self.l = l

        from sklearn.cluster import KMeans

        # Na teoria, precisamos apenas de 3 clusters,
        # na pratica, o dataset tem ruído e outliers,
        # então vale a pena buscar por variações do mesmo
        # padrão

        zx, mu, sd = zscore(x)

        for i in range(10):
            kmeans = KMeans(n_clusters=self.nlabels, n_init=3).fit(zx)

            # guarda o centroide de cada cluster encontrado
            self.cluster_centers = kmeans.cluster_centers_*sd + mu
            n_clusters = len(self.cluster_centers)
            # associa um label (squared, triangle...) a cada centroide
            # buscando qual é o mais comum
            self.cluster_labels = np.array([most_common(l[kmeans.labels_==i]) for i in range(n_clusters)])

            if len(set(self.cluster_labels)) >= len(shapes):
                break
        else:
            print("Não foi possível detectar todos os padrões no Kmeans.")

    def _dist_matrix(self, x, y):
        """Calcula a matriz de distância M x N entre os dados de treinamento
        e test usando o DTW

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Matriz de distância entre as séries x e y [training_n_samples, testing_n_samples]
        """

        dm_count = 0

        # Calcula a matriz de distancia condensada (triangulo superior) das distâncias entre x e y
        if(np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            p = ProgressBar(np.shape(dm)[0])

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = dtw(x[i, ::self.subsample_step], y[j, ::self.subsample_step], self.max_warping_window)

                    dm_count += 1
                    p.animate(dm_count)

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Calcula a matriz completa entre x e y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0]*y_s[0]

            p = ProgressBar(dm_size)

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = dtw(x[i, ::self.subsample_step], y[j, ::self.subsample_step], self.max_warping_window)
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)

            return dm

    def predict(self, x):
        """Infere a classe da série passada e a probabilidade dela

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array contendo o dado para teste

        Returns
        -------
          2 arrays :
              (1) a inferencia dos dados
              (2) a probabilidade dos pontos classificados
        """
        dm = self._dist_matrix(x, self.cluster_centers)

        # Identifica quais são as k soluções mais próximas
        knn_idx = dm.argsort()[:, :self.n_neighbors]
        knn_dists = dm[np.arange(dm.shape[0])[:, None], knn_idx]

        # Identifica os grupos pelo tipo (ex: square, triangle ...)
        knn_labels = self.cluster_labels[knn_idx]

        # Model Label
        # mode_data = mode(knn_labels, axis=1)
        # mode_label = mode_data[0]
        # mode_proba = mode_data[1]/self.n_neighbors
        #
        mode_calc = []
        for pattern, dists in zip(knn_labels, knn_dists):
            mode_calc.append([np.sum(dists[pattern == lb]) for lb in pattern])

        mode_data = np.array(mode_calc)
        mode_idx = np.argmin(mode_data, axis=1)
        mode_proba = 1 - mode_data[np.arange(mode_data.shape[0])[:, None], mode_idx]/np.sum(mode_data, axis=1)
        mode_label = knn_labels[np.arange(knn_labels.shape[0]), mode_idx]

        return mode_label.ravel(), mode_proba.ravel()

class ProgressBar:
    """Progress bar copiada do modulo PYMC
    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print('\r', self)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


shapes = ('square', 'triangle', 'hbar', 'vbar')

def load_data(freqs_dir=None):
    import glob

    if freqs_dir is None:
        FREQS_DATA_DIR = os.path.join('..', 'data', 'freqs_series')  # local dos arquivos
    else:
        if os.path.exists(freqs_dir):
            FREQS_DATA_DIR = freqs_dir
        else:
            raise Exception("Invalid data directory path")

    print(FREQS_DATA_DIR)


    lsamples = []
    class_ = []
    for shape in shapes:
        pattern = os.path.join(FREQS_DATA_DIR, '%s_*.dat' % shape)
        print(pattern)
        files = glob.glob(pattern)

        for i, filename in enumerate(files):
            sdf = pd.read_csv(filename, sep=',', header=None, names=['time', 'freq'])
            sample = np.zeros(3000)
            sample[sdf.time] = sdf.freq
            lsamples.append(sample)
            class_.append(shape)

    df = np.array(lsamples)
    cldf = np.array(class_)

    return cldf, df


def load_file(filename):
    sdf = pd.read_csv(filename, sep=',', header=None, names=['time', 'freq'])
    sample = np.zeros(3000)
    sample[sdf.time] = sdf.freq
    return sample


def align_data(df):

    # Remove o excesso de zeros

    start_j = 0
    print(df.shape)
    for j in range(df.shape[1]):
        if df[:,j].any():
            start_j = j-min(10, j)
            break

    end_j = 0
    for j in range(df.shape[1]-1, 0, -1):
        if df[:,j].any():
            end_j = j+min(10, df.shape[1]-j)
            break

    rdf = df[:, start_j:end_j]

    return rdf


def make_dataset(cldf, df, train_size=0.6):
    x_train = [] #df[chosen,:]
    y_train = [] #cldf[chosen]

    x_test = [] #df[test_chosen,:]
    y_test = [] #cldf[test_chosen]

    # Constrói os conjuntos de treinamento e test

    for shape in shapes:
        mask = cldf == shape
        indexes = np.arange(df.shape[0])[mask]
        if len(indexes) == 0:
            print("Pattern ", shape, " not present.")
            continue
        chosen = np.random.choice(indexes, replace=False, size=int(len(indexes)*train_size))
        test_chosen = [idx for idx in indexes if idx not in chosen]

        x_train.extend([df[i,:] for i in chosen])
        y_train.extend([cldf[i] for i in chosen])

        x_test.extend([df[i,:] for i in test_chosen])
        y_test.extend([cldf[i] for i in test_chosen])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # TODO verificar se vai deixar assim, une hbar e vbar no mesmo conjunto
    # pois eles estavam gerando muitos falsos positivos

    # if len(y_train) > 0:
    #     y_train[y_train == 'hbar'] = 'bar'
    #     y_train[y_train == 'vbar'] = 'bar'

    # if len(y_test) > 0:
    #     y_test[y_test == 'hbar'] = 'bar'
    #     y_test[y_test == 'vbar'] = 'bar'

    print(x_test.shape)

    return x_train, y_train, x_test, y_test


def report(y_test, label, step=1):

    print (classification_report(y_test[::step], label))

    used_labels = shapes #['square', 'triangle', 'bar']
    nused = len(used_labels)
    conf_mat = confusion_matrix(y_test[::step], label, labels=used_labels)

    fig = plt.figure(figsize=(nused,nused))
    width = np.shape(conf_mat)[1]
    height = np.shape(conf_mat)[0]

    res = plt.imshow(np.array(conf_mat), cmap=cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c>0:
                plt.text(j-.2, i+.1, c, fontsize=16)

    cb = fig.colorbar(res)
    plt.title('Confusion Matrix')
    _ = plt.xticks(range(nused), used_labels, rotation=90)
    _ = plt.yticks(range(nused), used_labels)
    plt.tight_layout()

    plt.savefig("confmat.png")
    plt.show()


def runtest(freqs_dir=None, window=None, step=1, kclusters=8, neighbors=3):
    from matplotlib import cm
    from sklearn.metrics import classification_report, confusion_matrix

    from datetime import datetime

    labels, data = load_data(freqs_dir)

    data = align_data(data)
    x_train, y_train, x_test, y_test = make_dataset(labels, data)

    mww = window or int(x_train.shape[1] * 0.1)  # regra classica, usar 10% do tamanho do sinal

    algo = KNN(n_neighbors=neighbors, max_warping_window=mww, nlabels=kclusters)
    algo.fit(x_train[::step], y_train[::step])

    # Envia os dados de teste para serem classificados
    label, proba = algo.predict(x_test[::step])

    df = pd.DataFrame(data=y_test, index=range(len(y_test)), columns=['y_label'])
    df.loc[:, 'y_pred'] = label

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    df.to_csv("results_%s.csv" % ts)

    report(y_test, label, step=step)


def run_one_test(freqs_dir, input, window=None, step=1, kclusters=8, neighbors=3):
    from matplotlib import cm
    from sklearn.metrics import classification_report, confusion_matrix

    from datetime import datetime

    filename, clabel = input.split(':')
    target_data = load_file(filename)

    labels, data = load_data(freqs_dir)

    data = align_data(np.vstack([data, target_data]))
    x_train, y_train, _, _ = make_dataset(labels, data[:-1,:], 1.)
    _, _, x_test, y_test  = make_dataset(np.array([clabel]), data[-1,np.newaxis], 0.)

    mww = window or int(x_train.shape[1] * 0.1)  # regra classica, usar 10% do tamanho do sinal

    algo = KNN(n_neighbors=neighbors, max_warping_window=mww, nlabels=kclusters)
    algo.fit(x_train[::step], y_train[::step])

    # Envia os dados de teste para serem classificados
    label, proba = algo.predict(x_test[::step])

    print(label, proba, "correct result is: ", clabel)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Processsa  arquivos da camada II-III do NeuronalSys.')
    parser.add_argument('-w','--window', type=int, default=None,
                        help='Tamanho da janela de deformação avaliada pelo DTW (warping window)')
    parser.add_argument('--load', type=str, default=None,
                        help='Carrega a analise da execução salva no arquivo informado.')
    parser.add_argument('--dir', type=str, default=None,
                    help='Diretório contendo os arquivos com as amostras para treinar e testar a rede (60% treino/40% teste).')
    parser.add_argument('-i', '--input', type=str, default=None, help='Informa o arquivo com o objeto único para predição.')
    parser.add_argument('-c', '--clusters', type=int, default=8, help='Número de clusters formados pelo K-means (default 8).')
    parser.add_argument('-n', '--neighbors', type=int, default=3, help='Número de vizinhos considerados pelo KNN (default 3).')
    parser.add_argument('-l', '--list-shapes', action='store_true', help='Lista o nome dos formatos suportados.')

    args = parser.parse_args()

    if args.list_shapes:
        print(shapes)
        exit(0)

    if args.load is None and args.dir is not None:
        if args.input is None:
            runtest(args.dir, window=args.window, step =1, kclusters=args.clusters, neighbors=args.neighbors)
        else:
            run_one_test(args.dir, args.input, window=args.window, step =1, kclusters=args.clusters, neighbors=args.neighbors)
    else:
        df = pd.read_csv(args.load)
        report(df.loc[:,'y_label'], df.loc[:,'y_pred'].T, step=1)

