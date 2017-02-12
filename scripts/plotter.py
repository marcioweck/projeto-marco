#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib.ticker as ticker
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

def draw_rect(ax, pos, a, b, fill=False, edge_color='none', bg_color='none'):
    ax.add_patch(
        patches.Rectangle(
            pos,        # (x,y)
            a,          # width
            b,          # height
            fill=fill,
            facecolor=bg_color,
            edgecolor=edge_color
        )
    )

    return ax

def draw_triangle(ax, pos, b, h, fill=False, edge_color='none'):
    ax.plot([pos[0], pos[0]+b], [pos[1], pos[1]], c=edge_color)
    ax.plot([pos[0], pos[0]+b*0.5], [pos[1], pos[1]-h], c=edge_color)
    ax.plot([pos[0]+b*0.5, pos[0]+b], [pos[1]-h, pos[1]], c=edge_color)
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
            return draw_rect(ax, (x,y), a, b, edge_color='blue')
        elif shape == 'triangle':
            return draw_triangle(ax, (x,y), a, b, edge_color='blue')
        else:
            print('Formato não encontrado.')
        
        return None
    except Exception as e:
        print("File cannot be parsed correctly.", e)


def plot_shape(filename, area="40x40"):
    a, b = parse_area(area)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    xticks = np.arange(0, 1, 1./a)
    yticks = np.arange(0, 1, 1./b)
    
    ax2 = draw_rect(ax1, (0,0), a/float(a),b/float(b), fill=True, bg_color='black')

    ax3 = parse_file(ax1, filename, (a,b))
    
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1./a))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(1./b))

    ax3.xaxis.set_ticklabels(xticks*a) 
    ax3.yaxis.set_ticklabels(yticks*b)
    ax3.xaxis.tick_top()
    ax3.invert_yaxis()    


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Desenha arquivos da camada II-III do NeuronalSys.')
    parser.add_argument('-a', '--area', type=str, default="40x40",
                        help='Tamanho da área de desenho. Exemplo: 40x40 representa uma área de plot de 40x40 pontos.')
    parser.add_argument('-f', '--file', required=True, type=str, help="Nome do arquivo.")

    args = parser.parse_args()

    _, filename = os.path.split(r""+args.file)

    plot_shape(filename, args.area)
    plt.show()

