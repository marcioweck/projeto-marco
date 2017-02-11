#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

import numpy as np


def parse_area(area_str):
    try:
        a_b = area_str.split('x')
        a = int(a_b[0])/100.
        b = int(a_b[1])/100.
    except:
        print('''Valor da area não foi corretamete inserido.
                 Valor %s, valor esperado (exemplo): 40x40.''' % area_str)
    
    return (a, b)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111, aspect='equal')
    # ax1.add_patch(
    #     patches.Rectangle(
    #         (0, 0),   # (x,y)
    #         a,          # width
    #         b,          # height
    #         fill=False
    #     )
    # )
    # plt.show()

def draw_rect(ax, pos, a, b, fill=False, color='none'):
    ax.add_patch(
        patches.Rectangle(
            pos,        # (x,y)
            a,          # width
            b,          # height
            fill=False,
            edgecolor=color
        )
    )

    return ax

def draw_triangle(ax, pos, b, h, fill=False, color='none'):
    ax.plot([pos[0], pos[0]], [pos[1], pos[1]+b], c='b')
    ax.plot([pos[0], pos[0]+h], [pos[1], pos[1]+b*0.5], 'b')
    ax.plot([pos[0]+h, pos[0]], [pos[1]+b*0.5, pos[1]], 'b')
    return ax


def parse_file(ax, filename, dimensions):
    # triangle_27x13_p9x22_209.dat
    try:
        shape, size, pos, _ = filename.split('_')
        a, b = parse_area(size)
        x, y = parse_area(pos[1:])

        a /= dimensions[0]
        x /= dimensions[0]
        b /= dimensions[1]
        y /= dimensions[1]

        print(shape, size, pos)
        if shape == 'square':
            return draw_rect(ax, (x,y), a, b, color='blue')
        elif shape == 'triangle':
            return draw_triangle(ax, (x,y), a, b, color='blue')
        else:
            print('Formato não encontrado.')
        
        return None
    except Exception as e:
        print("File cannot be parsed correctly.", e)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Desenha arquivos da camada II-III do NeuronalSys.')
    parser.add_argument('-a', '--area', type=str, default="40x40",
                        help='Tamanho da área de desenho. Exemplo: 40x40 representa uma área de plot de 40x40 pontos.')
    parser.add_argument('-f', '--file', required=True, type=str, help="Nome do arquivo.")

    args = parser.parse_args()

    a, b = parse_area(args.area)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    ax2 = draw_rect(ax1, (0,0), a/float(a),a/float(a))

    ax3 = parse_file(ax2, args.file, (a,b))
                                      
    Y, X = np.mgrid[0:1:a/100., 0:1:b/100.]
    
    plt.scatter(X, Y, c='k', alpha=0.7)

    plt.show()




