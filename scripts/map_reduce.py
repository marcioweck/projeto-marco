"""
Script to reduce a raw activation map into activation frequency time serie
"""

import pandas as pd
import csv


shapes = [
    'square',
    'rectangle',
    'triangle',
    'hbar',
    'vbar',
    'circle'
]


def choose_name(header, seqnum):
    """
    @brief      Formata o nome do arquivo para ser mais descritivo

    @param      header  Metadados obtidos na primeira linha do arquivo
    @param      seqnum  Identificador global do arquivo

    @return     Novo nome do arquivo de frequências
    """
    name = "{}_{}x{}_p{}x{}_{}.dat".format(shapes[int(header['type'])-1],
                                         int(header['n']), int(header['m']),
                                         int(header['y']), int(header['x']), seqnum)
    return name


if __name__ == '__main__':
    """
    Busca em todos os arquivos da pasta passada como primeiro parâmetro e conta
    quantos neurônios ativaram naquele instante de tempo.
    """
    import sys
    import os
    import glob

    if len(sys.argv) < 2:
        raise Exception("Usage: python map_reduce.py <data_dir_path>")
    data_dir = sys.argv[1]
    out_data_dir = sys.argv[2]
    files = glob.glob(os.path.join(data_dir, '*.txt'))

    total = len(files)
    for i, filename in enumerate(files):
        print("Reading ", filename, '::', i, 'of', total)

        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=['x', 'y', 'n', 'm', 'type'])
            header = next(reader)

        df = pd.read_csv(filename, sep=',', skiprows=[0], header=None,
                         names=['col', 'row', 'time'])

        rdf = df.time.value_counts(sort=False)
        rdf.to_csv(os.path.join(out_data_dir, choose_name(header, i)), sep=',')
