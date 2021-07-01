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
        if -0.0000001 < resu < 0.0000001:
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
            Fx.append(funciones[i])
            vari.append(je)
            dxQ.append(dx[i][j])
    #iterar para encontrar las raices
    while fin == 1:
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
        #Sacanod las h's
        hs = dxevals.dot(z)
        #Calculando las nuevas "x"
        vec=np.add(hs,vec)
        results.append(vec)

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

def InterPol(x,y,color=['ro','b'], val=0, polyrange=[0,5]):
    '''Esta función sirve para resolver por medio del método de Newton-Raphson Multivariable.
   Necesita como entrada las funciones y los valores  inicial X0.
   Ejemplo de uso:
   ## Valores a interpolar##
   x=[930,983,1050,1088,1142,1316,1320]
   y=[10,20,40,60,100,200,400]
   ## Función ##
   InterPol(x,y,val=1250,color=['gd','m--'],polyrange=[0,5])
   '''
    x=np.array(x)
    y=np.array(y)
    error=[]
    a={}
    for i in range(polyrange[0],polyrange[1]):
        coeficientes = np.polyfit(x, y,i+1)
        evaluacion = np.polyval(coeficientes,x)
#         error=((1/len(x))* np.sum(np.subtract(y,evaluacion)**2))
        error = np.sum(np.subtract(y,evaluacion))
        a[error]=coeficientes
    mejor = min(a.keys())
    poly = np.poly1d(a[mejor])
    new_x = np.linspace(x[0], x[-1])
    new_y = poly(new_x)
    plt.plot(x, y, color[0], new_x, new_y,color[1])
    plt.xlim([x[0]-1, x[-1] + 1 ])
    print('Polinómio: \n',poly)
    if val != 0:
        calculado=np.polyval(a[mejor],val)
        print('\n El valor para {} es {}'.format(val,calculado))
    return

def intercalar_mm(xs,y, valor):
  xs = np.array(xs)
  y = np.array(y)
  mayores = xs[xs > valor][:]
  menores = xs[xs[xs < valor].argsort()][::-1]
  mayoresy = y[np.where(xs>valor)]
  menoresy = y[np.where(xs<valor)][::-1]
  traslp = []
  traslpy = []
  i=0
  while i <= np.abs(len(mayores)-len(menores))+2 :
    while i < (min(len(menores),len(mayores))):
      traslp.append(mayores[i])
      traslp.append(menores[i])
      traslpy.append(mayoresy[i])
      traslpy.append(menoresy[i])
      i += 1
    if i*2 == len(xs):
      if len(traslp) == len(xs):
        return traslp, traslpy

    if len(mayores) ==  (min(len(menores),len(mayores))):
      traslp.append(menores[i])
      traslpy.append(menoresy[i])
      if len(traslp) == len(xs):
        return traslp, traslpy

    else:
      traslp.append(mayores[i])
      traslpy.append(mayoresy[i])
      if len(traslp) == len(xs):
        return traslp, traslpy
    i += 1
  return traslp, traslpy

def InterPol_Intercal(x,y,color=['ro','b'], val=0, polyrange=[0,5], gen = 0, mgraphs = 0):
  '''Esta función sirve para realizar un modelo que ajuste a los datos por medio
  del método de interpolación. Hace el ajuste de acuerdo a la población más densa
  de datos. Recibe como entrada vectores "x" y "y". Y si es requerido un valor
  específico a evaluar. Puedes jugar con los parámetros para que se ajuste con distintas
  métricas de error mediante el parámetro "gen", distintos números de gráficas con mgraphs.
  Ejemplo de uso:
  ## Valores a interpolar##
  x=[930,983,1050,1088,1142,1316,1320]
  y=[10,20,40,60,100,200,400]
  ## Función ##
  InterPol_Intercal(x,y,color=['kD','c--'], val=1250, polyrange=[1,7], gen = 1, mgraphs=0)
  '''
  xr,yr = x,y
  x,y = intercalar_mm(x,y,val)
  error=[]
  evaluacion = []
  a={}

  if mgraphs:
    fig, axs = plt.subplots(int(np.floor(polyrange[1]/2)),2, figsize=(15,15))
    fig.suptitle('Puntos ajustados distintos grados de polinomio')

  for k in range(polyrange[0],polyrange[1]):
    ecs=np.zeros([k+1,k+1])
    for i in range(k+1):
        for j in range(k+1):
            x1_eval=x[i]**j
            ecs[i,j]=x1_eval


    coeficientes=np.linalg.solve(ecs,y[:k+1])
    polyeval = 0
    evaluacion = np.polyval(coeficientes[::-1],x[:k+1])

    if gen == 1:
      error = np.sum(np.subtract(y[:k+1],evaluacion))
    else:
      error = np.abs(np.sum(np.subtract(y[:k+1],evaluacion)))

    a[error]=coeficientes

    if mgraphs:
      poly = np.poly1d(coeficientes[::-1])
      new_x = np.linspace(x[0], x[-1])
      new_y = poly(new_x)

      kr = k -1
      axs[int(np.floor(kr/2)), kr%2].plot(x, y, "kD", new_x, new_y,'c--')
      axs[int(np.floor(kr/2)), kr%2].set_title('Poliniomio de grado: {}'.format(k))

  mejor = min(a.keys())
  poly = np.poly1d(a[mejor][::-1])
  if not mgraphs:
    # poly = np.poly1d(a[mejor][::-1])
    new_x = np.linspace(xr[0], xr[-1])
    new_y = poly(new_x)
    plt.plot(xr, yr, color[0], new_x, new_y,color[1])
    # plt.xlim([x[0]-1, x[k] + 1 ])
  print('Polinómio: \n',poly)
  if val != 0:
      calculado=np.polyval(a[mejor][::-1],val)
      print('\n El valor con polinomio de grado {} para {} es {}'.format( len(a[mejor])-1,val,calculado))

def Lagrange(x,y,val=0,orden=0):
    '''Rubs, si quieres calcular un grado menor a dos, mejor calculalo a partir de 3 y tendrás los que necesite.'''
    xg, yg = x,y
    x,y = intercalar_mm(x,y,val)
    X = Symbol('X')
    if orden:
        orden
    else:
        orden = len(x)-1
    if mgraphs:
        fig, axs = plt.subplots(int(np.ceil(orden/2)),2, figsize=(15,15))
        fig.suptitle('Puntos ajustados distintos grados de polinomio')

    for k in range(1,orden+1):
        coef=[]
        Li,Fx=1,0
        for i in range(k+1):
            Li=1
            for j in range(k+1):
                if i != j:
                    Li*=(X-x[j])/(x[i]-x[j])
            Fx+=Li*y[i]
        for m in range(k+1):
            coef.append(simplify(collect(Fx,X)).coeff(X,m))
    #         print(coef)
        Fxe = Fx.evalf(subs={X:val})
        print('Polinomio de Lagrange de orden {}: {}'.format(k, simplify(Fx)))
        print('\t Resultado con el polinomio de Lagrange de orden {}:  {}'.format(k,Fxe))
        if mgraphs:
            poly = np.poly1d(coef[::-1])
            new_x = np.linspace(xg[0], xg[-1])
            new_y = poly(new_x)

            kr = k -1
            axs[int(np.floor(kr/2)), kr%2].plot(xg, yg, "kD", new_x, new_y,'c--')
            axs[int(np.floor(kr/2)), kr%2].set_title('Poliniomio de grado: {}'.format(k))

def minimos_cuadrados_uni(x,y,grado=1):
    '''Ejemplo:
    gr = np.array([0,1,2,3,4,5,6,7])
    hr = np.array([12,10.5,10,8,7,7.5,8.5,9])
    minimos_cuadrados_uni(gr,hr,grado=2)'''
    x=np.array(x)
    y=np.array(y)
    np.set_printoptions(suppress=True)
    n=len(x)
    cont=0
    m = np.zeros([grado+1,grado+1])
    s= []
    m[0,0] = n
    for j in range(grado+1):
        s.append(np.sum(np.power(x,j)*y))
    for i in range(grado+1):
        for j in range(grado+1):
            if i != 0 or j!=0:
                m[i,j]=np.sum(np.power(x,cont+1))
                cont+=1
        cont-=grado
    sistema=np.linalg.solve(m,s)
    print('La matriz de sumas es: \n',m)
    print('\nEl vector de solución es: \n',s)
    for i in range(grado+1):
        print('\nEl coeficiente a{} es: {}\n'.format(i,sistema[i]))
    st=np.sum(np.power(y-np.mean(y),2))
    c = sistema[0]
    for k,j in enumerate(sistema):
        if k!=0:
            c+=j*np.power(np.array(x),k)
    sr=np.sum(np.power(y-c,2))
    r2 =(st-sr)/st
    r=np.sqrt((st-sr)/st)
    print('Coeficiente de determinación: ',r2,'\n')
    print('Coeficiente de correlación: ',r,'\n')
    print('Porcentaje que explica el modelo de la incertidumbre original: ', np.round(r*100,3),'%')

def minimos_cuadrados_multi(vec,y,eval=0):
    '''Ejemplo:
    vec=[[0.02,0.02,0.02,.02,.1,.1,.1,.1,.18,.18,.18,.18],[1000,1100,1200,1300,1000,1100,1200,1300,1000,1100,1200,1300]]
    y=[78.9,65.1,55.2,56.4,80.9,69.7,57.4,55.4,85.3,71.8,60.7,58.9]
    minimos_cuadrados_multi(vec,y,eval=1)'''
    np.set_printoptions(suppress=True)
    y=np.array(y)
    grado = len(vec)
    n=len(y)
    cont=0
    x={}
    sumas={}
    for i in range(len(vec)):
        x[i]=np.array(vec[i])
    m = np.zeros([grado+1,grado+1])
    s= []
    s.append(np.sum(y))
    for j in range(grado):
        s.append(np.sum(x[j]*y))
    for i in range(1,grado+1):
        m[0,i]=np.sum(x[i-1])
        m[i,0]=np.sum(x[i-1])
    for i in range(1,grado+1):
        for j in range(1,grado+1):
            m[i,j]=np.sum(x[i-1]*x[j-1])
    m[0,0] = n
    sistema=np.linalg.solve(m,s)
    print('La matriz de sumas es: \n',m)
    print('\nEl vector de solución es: \n',s)
    for i in range(grado+1):
        print('\nEl coeficiente a{} es: {}\n'.format(i,sistema[i]))
    st=np.sum(np.power(y-np.mean(y),2))
    c = sistema[0]
    for k,j in enumerate(sistema):
        if k!=0:
            c+=j*np.array(vec[k-1])
    sr=np.sum(np.power(y-c,2))
    r2 =(st-sr)/st
    r=np.sqrt((st-sr)/st)
    print('Coeficiente de determinación: ',r2,'\n')
    print('Coeficiente de correlación: ',r,'\n')
    print('Porcentaje que explica el modelo de la incertidumbre original: ', np.round(r*100,3),'%')
    if eval:
        print('\nLos valores calculados son:')
        for i in range(len(vec[0])):
            e = sistema[0]
            for k,j in enumerate(sistema):
                if k!=0:
                    e+=j*vec[k-1][i]
            print(e)

def mnc_sinusoidal(x,y,periodo,eval=1):
    '''Ejemplo:
    y = np.array([7.3,7,7.1,6.5,7.4,7.2,8.9,8.8,8.9,7.9,7])
    x= np.array([0,2,4,5,7,9,12,15,20,22,24])
    periodo=24
    mnc_sinusoidal(x,y,periodo,eval=1)'''
    np.set_printoptions(suppress=True)
    w0 = (2*np.pi)/periodo
    n=len(x)
    sy=np.sum(y)
    scos = np.sum(np.cos(w0*x))
    ssen = np.sum(np.sin(w0*x))
    scos2 = np.sum(np.power(np.cos(w0*x),2))
    scossen = np.sum(np.cos(w0*x)*np.sin(w0*x))
    ssen2 = np.sum(np.power(np.sin(w0*x),2))
    sycos = np.sum(np.cos(w0*x)*y)
    sysen = np.sum(np.sin(w0*x)*y)
    sy,w0,scos,ssen,scos2,scossen,ssen2,sycos,sysen
    fila1=[n,scos,ssen]
    fila2=[scos,scos2,scossen]
    fila3=[ssen,scossen,ssen2]
    sols=[sy,sycos,sysen]
    c=np.linalg.solve([fila1,fila2,fila3],sols)
    print('La matriz de sumas es: \n',np.array([fila1,fila2,fila3]))
    print('\nEl vector de solución es: \n',sols)
    print('\nParámetros de a0, a1, b1: \n')
    print('A0: ',c[0], '\t\tA1: ',c[1], '\tB1: ',c[2])
    C=np.sqrt(c[1]**2+c[2]**2)
    theta = np.arctan(-c[2]/c[1])
    print('\n Valor medio: {}\n Amplitud oscilatoria: {}\n Ángulo: {}\n'.format(c[0],C,theta))
    st=np.sum(np.power(y-np.mean(y),2))
    sr=np.sum(np.power(y-(c[0]+c[1]*np.cos(w0*x)+c[2]*np.sin(w0*x)),2))
    r2 =(st-sr)/st
    r=np.sqrt((st-sr)/st)
    print('Coeficiente de determinación: ',r2,'\n')
    print('Coeficiente de correlación: ',r,'\n')
    print('Porcentaje que explica el modelo de la incertidumbre original: ', np.round(r*100,3),'%\n')
    if eval:
        print('Los resultados calculados son: \n',(c[0]+c[1]*np.cos(w0*x)+c[2]*np.sin(w0*x))[np.newaxis, :].T)
