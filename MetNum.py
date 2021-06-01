from sympy import *
import math
import numpy as np
import pandas as pd


def MN_PF(funcion,despeje, X0):
    '''Esta función sirve para resolver por medio del punto fijo.
       Ayudará a Rubs a hacer los ejercicios rapidamente.
       Necesita entrar la función original y el despeje de dicha función,
       así como el valor X0 inicial.
       Ejemplo de uso:
       X0 = -5
       X = Symbol('X')
       funcion = cos(X)-X**2
       despeje = cos(X)**(1/2)
       MN_PF(funcion,despeje, X0)'''
    fin = 1
    gxs = []
    results = [] 
    while fin == 1:
      X0 = despeje.evalf(subs={X: X0})
      resu = funcion.evalf(subs={X: X0}) 
      gxs.append(X0)
      results.append(resu) 

      if -0.00001 < resu < 0.00001:
        fin =0

    allres = { 'g(x)': gxs,
               'f(x)': results}

    allresdf = pd.DataFrame(allres)
    return allresdf

def MN_PFM(despejex,despejey, mode='xy'):
    '''Esta función sirve para resolver por medio del punto fijo multivariable.
       Funciona únicamente con dos variables.
       Necesita como entrada la función de X y la de Y, ya despejadas.
       Opcionalmente necesita el modo de operación. Hay dos: 'xy' o 'yx'.
       Ejemplo de uso:
       X = Symbol('X')
       Y = Symbol('Y')
       despejex = ((X**2)-Y)/(5*Y)
       despejey = -(X**2)+X+0.75
       MN_PFM(despejex,despejey,mode='xy')'''
    X0 = 0
    Y0 = 0
    fin = 1
    gxs = []
    gys = [] 
    if mode == 'xy':
        while fin == 1:
          xa = X0
          ya = Y0
          X0 = despejex.evalf(subs={X: X0, Y: Y0})
          Y0 = despejey.evalf(subs={X: X0, Y: Y0})
          resx = xa - X0
          resy = ya - Y0
          gxs.append(X0)
          gys.append(Y0)

          if -0.00000001 < resx < 0.00000001 and -0.00000001 < resy < 0.00000001:
            fin =0

        allres = { 'X': gxs,
                   'Y': gys}
    elif mode == 'yx':
        while fin == 1:
          xa = X0
          ya = Y0
          Y0 = despejey.evalf(subs={X: X0, Y: Y0})
          X0 = despejex.evalf(subs={X: X0, Y: Y0}) 
          resx = xa - X0
          resy = ya - Y0
          gxs.append(X0)
          gys.append(Y0)

          if -0.00000001 < resx < 0.00000001 and -0.00000001 < resy < 0.00000001:
            fin =0

        allres = { 'X': gxs,
                   'Y': gys}
    else:
        raise Exception('Escoge mode xy o yx')
        

    allresdf = pd.DataFrame(allres)
    return allresdf

def MN_NR(funcion, X0):
    '''Esta función sirve para resolver por medio del método de Newton-Raphson.
       Necesita como entrada la función y el valor inicial X0.
       Ejemplo de uso:
       X0 = 2.86      # Valor inicial
       ### Constantes de la función ###
       P = 10
       T = 353.15
       a =  3.599
       b = 0.04267
       R = 0.08205
       ### Variable de la función ###
       X = Symbol('X')
       funcion = (P*(X**3))-(P*b+R*T)*(X**2)+(a*X)-(a*b)
       MN_NR(funcion, X0)'''
    dx = diff(funcion,X)
    dx = simplify(dx)
    print(dx)
    gxs = []
    results = [] 
    fin = 1
    while fin == 1:
        X0 = X0 - (funcion.evalf(subs={X: X0}) / dx.evalf(subs={X: X0}))
        resu =  funcion.evalf(subs={X: X0}) 
        print(X0)
        gxs.append(X0)
        results.append(resu) 
        if -0.0000001 < resu < 0.0000001:
            fin = 0 

    allres = { 'x_i': gxs,
                'f(x)': results}

    allresdf = pd.DataFrame(allres)
    return allresdf