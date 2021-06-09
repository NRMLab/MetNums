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
    X = Symbol('X')
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

def MN_PFM(despejes, vec, variables):
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


    fin = 1
    results = []
    while fin == 1:

        veca = vec.copy()

        for i,j in enumerate(despejes):
            d=dict(zip(variables,vec))
            vec[i]=j.evalf(subs=d)
        results.append(vec.copy())

        res = np.subtract(veca,vec)
        cont = 0
        for i in res:
            if -0.000000000001 < i < 0.000000000001:
                cont+=1
        if cont == len(despejes):
            fin=0
        else:
            cont=0
    varn=[]
    for i,j in enumerate(variables): 
        varn.append(j.name)

    allresdf = pd.DataFrame(results)
    allresdf.columns=varn
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
    X = Symbol('X')
    dx = diff(funcion,X)
    dx = simplify(dx)
    print(dx)
    gxs = []
    results = [] 
    fin = 1
    while fin == 1:
        X0 = X0 - (funcion.evalf(subs={X: X0}) / dx.evalf(subs={X: X0}))
        resu =  funcion.evalf(subs={X: X0}) 
#         print(X0)
        gxs.append(X0)
        results.append(resu) 
        if -0.000000001 < resu < 0.000000001:
            fin = 0 

    allres = { 'x_i': gxs,
                'f(x)': results}

    allresdf = pd.DataFrame(allres)
    return allresdf

##Metodo Newton-Rapshon Multivariable##
def MN_NRM(funciones, vec, variables):
    '''Esta función sirve para resolver por medio del método de Newton-Raphson Multivariable.
       Necesita como entrada las funciones y los valores  inicial X0.
       Ejemplo de uso:
       ## Valores iniciales ##
       vec=[1,2,3]
       ## Costantes de la función ##
       a = 1
       b = 2
       c = 3
       ## Variables de la función ##
       X = Symbol('X')
       Y = Symbol('Y')
       Z = Symbol('Z')
       F1 = x1**2+3 (2/x) + X2*3 +X3
       F2 = X2**2+4+X3+X2**9 + X1**3
       F3 = X1**3+X2+X3**2
       funciones=[F1,F2,F3]
       variables=[X,Y,Z]
       MN_NRM(funciones, vec, variables)
       ''' 
    Fx=[]
    vari=[]
    dxQ=[]
    fin = 1
    results=[]
    d=dict(zip(variables,vec))
    LF=len(funciones)
    dx=np.zeros([LF,len(variables)])  # Crea una matriz de cerps del tamaño de la Jacobiana
    dx=np.ndarray.tolist(dx)
    dxeval=np.zeros([LF,len(variables)]) #Crea una matriz para las diferenciales evaluadas

    funceval=np.zeros([LF,1])
    #Sacar las derivadas 
    for i in range(LF):
        for j,je in enumerate(variables):

            dx[i][j]=diff(funciones[i],je)
            print(dx[i][j])
            Fx.append(funciones[i])
            vari.append(je)
            dxQ.append(dx[i][j])
            
    c1 = 0
    #iterar para encontrar las raices 
    while fin == 1:
        c1 += 1
        #Hacer las evaluaciones
        for i in range(LF):
            #evaluar funciones
            funceval[i]=funciones[i].evalf(subs=d)
            for j,je in enumerate(variables):
                #Evaluar las derivadas
                dxeval[i][j]=dx[i][j].evalf(subs=d)
        
        #Sacando la matriz inversa de las Jacobiana
        dxevals=np.linalg.inv(dxeval)
        #Multiplicar por -1 las funciones
        z= np.array(funceval).dot([[-1]])
        #Pasado de ser un ser 2D a 1D
        z=np.squeeze(z.T)
        #Sacando las h's
        hs = dxevals.dot(z) 
        #Calculando las nuevas "x"
        vec=np.add(hs,vec)
        results.append(vec)
        print(c1,'\t x: {0:.6f}'.format(vec[0]), ' y: {0:.6f}'.format(vec[1]))
        d=dict(zip(variables,vec))
        #Criterio de paro 
        cont=0

        for i in funceval:
            if -0.000000000001 < i < 0.000000000001:
                cont+=1
        if cont == LF:
            fin=0
        else:
            cont=0
        varn=[]
        for i,j in enumerate(variables): 
            varn.append(j.name)
    allresdf = pd.DataFrame(results)
    allresdf.columns=varn
    m=Matrix(dx)

    tabledxprint=pd.DataFrame([Fx,vari,dxQ]).T
    tabledxprint.columns=["Funciones","Variables","Derivadas"]

    return allresdf,  tabledxprint, m
#Rubs Estuvo Aqui jeje
#Otro cambio