# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 20:01:08 2021

@author: romain
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, AutoLocator)
from lmfit import Model

#analyse de l'image/analyse of picture

def lectureimage(path):
    img = cv2.imread(path+".tif", 1)
    plt.imshow(img)
    plt.show()
    height, width, c = img.shape
    newimg = np.zeros((height, width, 3), dtype = "uint8")
    for i in range(0,height):
        for j in range(0,width):#width
            value = img[i, j]
            newimg[i, j] = value
            #Gris = img[i,j]
            #print(i,j)
    #plt.imshow(newimg)
    plt.show()
    
    return height, width , newimg

def ZoomImage(img,xmin,xmax,ymin,ymax):
    imgZoom = np.zeros((ymax-ymin, xmax-xmin, 3), dtype = "uint8")
    for i in range(ymin,ymax):
        for j in range(xmin,xmax):#width
            value = img[i, j]
            imgZoom[i-ymin, j-xmin] = value     
    plt.imshow(imgZoom)
    plt.show()
    
    return imgZoom

def miseXgraph(img,xmin,xmax,ymin,ymax):
    xgraph = []
    Indicex = []
    Moyy = 0
    Value = 0
    for i in range(xmin,xmax):
        for j in range(ymin,ymax):
            value = img[j, i]
            Value = Value + value[0] + value[1] + value[2]
        Moyy = Value/(3*xmax)
        Indicex.append(i)
        xgraph.append(Moyy)
        Value = 0
    return xgraph , Indicex

def miseYgraph(img,xmin,xmax,ymin,ymax):
    ygraph = []
    Indicey = []
    Moyy = 0
    Value = 0
    height, width, c = img.shape
    for i in range(0,height):
        for j in range(0,width):
            value = img[i, j]
            Value = Value + value[0] + value[1] + value[2]
        Moyy = Value/(3*xmax)
        Indicey.append(i)
        ygraph.append(int(Moyy))
        Value = 0
    return ygraph , Indicey

def corcetYgraphique(Indicey,ygraph):
    newvalT = []
    for i in range(len(Indicey)):
        if ygraph[i] == 0 and Indicey[i] != Indicey[0] and Indicey[i] != Indicey[-1]:
            newval = (ygraph[i+1]+ygraph[i-1])/2
            newvalT.append(newval)
        else:
            newvalT.append(ygraph[i])
    valeur_max = newvalT[0]
    for j in range(len(newvalT)):
        if valeur_max < newvalT[j]:
            valeur_max = newvalT[j]
            Indice_max = j
    return newvalT , Indice_max


#analyse de la tache focal/analyse of focal spot

def Gauss(x, A, u, sig):
    return ( A * (np.exp((-(x-u)**(2))/(2.0*(sig**(2))))))

def Poly(x, a, b, c):
    return a * x**2 + b * x + c

def Gaussianoptics(z,wo,lamb,u):
    return wo * (1+((z-u)/((np.pi*wo**2)/lamb))**2)**(1/2)

def optiGx(u,sig, X, nbr):
    cont = 0
    for i in range(len(X)):
        if X[i] < u - nbr*sig :
            xmin, xminIndice = X[i] , i
        if X[i] > u + nbr*sig and cont == 0 :
            cont = 1
            xmax, xmaxIndice = X[i] , i
    return xmin, xminIndice, xmax ,xmaxIndice

def optiGy(u,sig, Y, nbr):
    cont = 0
    for i in range(len(Y)):
        if Y[i] < u - nbr*sig :
            ymin, yminIndice = Y[i] , i
        if Y[i] > u + nbr*sig and cont == 0 :
            cont = 1
            ymax, ymaxIndice = Y[i] , i
    return ymin, yminIndice, ymax , ymaxIndice

def recalcul(xT, yT, xminIndice, xmaxIndice):
    xreval = 0
    yreval = 0
    xrevalT = []
    yrevalT = []
    for i in range(len(xT)):
        if xT[i] >= xT[xminIndice] and xT[i] <= xT[xmaxIndice]:
            xreval = xT[i]
            yreval = yT[i]
            xrevalT.append(xreval)
            yrevalT.append(yreval)
    return xrevalT, yrevalT


#automatisation de la lecture du fichier d'entrée/automation reading of the input file

def lectureauto(path):
    titlex = []
    titley = []
    titlez = []
    namefichier = []
    positionvernier = []
    DO = []
    file = open(path+'.csv' , "r")
    a = 0    
    for ligne in file:
        mots= ligne.split(";")
        if a == 0:
            try :
                x=mots[0]
                y=mots[1]
                z=mots[2]
                titlex.append(x)
                titley.append(y)
                titlez.append(z)
                a = a + 1
            except:
                a = a + 1
        elif a > 0:
            try:
                x=mots[0]
                y=mots[1]
                z=mots[2]
                y=y.replace(",",".")
                z=z.replace(",",".")
                x = str(x)
                y = float(y)
                z = float(z)
                namefichier.append(x)
                positionvernier.append(y)
                DO.append(z)
            except:
                a = a + 1
    file.close()
    
    return namefichier, positionvernier, DO

#création du main/create main

def main(path):
    
    try:
        height, width, newimg = lectureimage(path)

        xmin = 0
        xmax = width
        ymin = 0
        ymax = height

        #localisation de la tache focale/location of the focal spot

        XT, Indicex = miseXgraph(newimg,xmin,xmax,ymin,ymax)

        YT, Indicey = miseYgraph(newimg,xmin,xmax,ymin,ymax)
    
        newYT = corcetYgraphique(Indicey,YT) #correction instrumentale CCD

        Gmodel = Model(Gauss,independent_vars=['x'])
        params = Gmodel.make_params()
        params['A'].set(50,vary= True#, min=1, max = 30000000
                        )
        params['u'].set(width/2)
        params['sig'].set(50,vary = True, min=0, max = 3000)

        resultGx = Gmodel.fit(XT, params ,x=Indicex )

        #print(resultGx.fit_report())
        #print('---------------------------------------------------------------------')
    
        xmin, xminIndice, xmax, xmaxIndice = optiGx(resultGx.params['u'].value, resultGx.params['sig'].value, Indicex, 3)
        imgZoom = ZoomImage(newimg,xmin,xmax,ymin,ymax)
        Indicexsig, XTsig = recalcul(Indicex, XT, xminIndice, xmaxIndice)



        YT, Indicey = miseYgraph(imgZoom,xmin,xmax,ymin,ymax)
        newYT, Indice_max = corcetYgraphique(Indicey,YT)
    

        Gmodel = Model(Gauss,independent_vars=['x'])
        params = Gmodel.make_params()
        params['A'].set(50
                        ,vary= True#, min=1, max = 30000000
                        )
        params['u'].set(Indice_max)
        params['sig'].set(50,vary = True, min=0, max = 3000)

        resultGy = Gmodel.fit(newYT, params ,x=Indicey )

        #print(resultGy.fit_report())
        #print('---------------------------------------------------------------------')

        ymin, yminIndice, ymax, ymaxIndice = optiGy(resultGy.params['u'].value, resultGy.params['sig'].value, Indicex, 3)
        imgZoom = ZoomImage(newimg,xmin,xmax,ymin,ymax)
        Indiceysig, YTsig = recalcul(Indicey, newYT, yminIndice, ymaxIndice)
        
        ymin = int(resultGy.params['u'].value - 2)
        ymax = int(resultGy.params['u'].value + 2)
        
        newXT, newIndicex = miseXgraph(newimg,xmin,xmax,ymin,ymax)

        Gmodel = Model(Gauss,independent_vars=['x'])
        params = Gmodel.make_params()
        params['A'].set(50
                        ,vary= True#, min=1, max = 30000000
                        )
        params['u'].set(resultGx.params['u'].value)
        params['sig'].set(50,vary = True, min=0, max = 3000)

        resultGxnew = Gmodel.fit(newXT, params ,x=newIndicex )

        #print(resultGxnew.fit_report())
        #print('---------------------------------------------------------------------')
        
        #plotting location focale spot
        
        '''
        fig, ax = plt.subplots(num = 1, figsize=(9,6),dpi=200)

        plt.plot(Indicex, XT,'ko')
        plt.plot(Indicex, resultGx.best_fit,'r-',label='fit ampl = %.1f , $\mu$ = %.2f , $\sigma$ = %.2f'%(resultGx.params['A'].value,resultGx.params['u'].value,resultGx.params['sig'].value),ms = 1.5)

        ax.xaxis.set_major_locator(AutoLocator())
        ax.xaxis.set_major_formatter('{x:.0f}')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.xlabel(r'X [pixel]',fontsize = 15)
        plt.ylabel(r"Moyenne d'intensité",fontsize = 15)
        plt.grid(True,'major','both',lw = 1.5)
        plt.grid(True,'minor','both',lw = 0.8)

        plt.legend(loc=0, fontsize = 9)
        plt.show()


        fig, ax = plt.subplots(num = 1, figsize=(9,6),dpi=200)
    
        #plt.plot(YT,Indicey,'ko')
        plt.plot(newYT,Indicey,'ko')
        plt.plot(resultGy.best_fit, Indicey,'r-',label='fit ampl = %.1f , $\mu$ = %.2f , $\sigma$ = %.2f'%(resultGy.params['A'].value,resultGy.params['u'].value,resultGy.params['sig'].value),ms = 1.5)
    
    
        #plt.plot(newYT,Indicey,'ko')
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(AutoLocator())
        ax.xaxis.set_major_formatter('{x:.0f}')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.xlabel(r"Moyenne d'intensité",fontsize = 15)
        plt.ylabel(r"Y [pixel]",fontsize = 15)
        plt.grid(True,'major','both',lw = 1.5)
        plt.grid(True,'minor','both',lw = 0.8)

        plt.legend(loc=0, fontsize = 9)
        plt.show()

        '''
        fig, ax = plt.subplots(num = 1, figsize=(9,6),dpi=200)

        plt.plot(newIndicex, newXT,'ko')
        plt.plot(newIndicex, resultGxnew.best_fit,'r-',label='fit ampl = %.1f , $\mu$ = %.2f , $\sigma$ = %.2f'%(resultGxnew.params['A'].value,resultGxnew.params['u'].value,resultGxnew.params['sig'].value),ms = 1.5)

        ax.xaxis.set_major_locator(AutoLocator())
        ax.xaxis.set_major_formatter('{x:.0f}')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.xlabel(r'X [pixel]',fontsize = 15)
        plt.ylabel(r"Moyenne d'intensité",fontsize = 15)
        plt.grid(True,'major','both',lw = 1.5)
        plt.grid(True,'minor','both',lw = 0.8)

        plt.legend(loc=0, fontsize = 9)
        plt.show()
        
        print('tache focal trouvé pour %s avec sigx = %.2f et sigy = %.2f'%(path,resultGxnew.params['sig'].value , resultGy.params['sig'].value))
        
        return resultGxnew.params['sig'].value , resultGy.params['sig'].value
    except:
        print('tache focale non-trouvé pour %s try new method'%(path))
        try:
        
            height, width, newimg = lectureimage(path)

            xmin = 350
            xmax = 550
            ymin = 250
            ymax = 450



            XT, Indicex = miseXgraph(newimg,xmin,xmax,ymin,ymax)

            YT, Indicey = miseYgraph(newimg,xmin,xmax,ymin,ymax)
    
            newYT = corcetYgraphique(Indicey,YT)

            Gmodel = Model(Gauss,independent_vars=['x'])
            params = Gmodel.make_params()
            params['A'].set(50,vary= True#, min=1, max = 30000000
                        )
            params['u'].set(width/2)
            params['sig'].set(50,vary = True, min=0, max = 3000)

            resultGx = Gmodel.fit(XT, params ,x=Indicex )

            #print(resultGx.fit_report())
            #print('---------------------------------------------------------------------')
            xmin, xminIndice, xmax, xmaxIndice = optiGx(resultGx.params['u'].value, resultGx.params['sig'].value, Indicex, 3)
            imgZoom = ZoomImage(newimg,xmin,xmax,ymin,ymax)
            Indicexsig, XTsig = recalcul(Indicex, XT, xminIndice, xmaxIndice)


            YT, Indicey = miseYgraph(imgZoom,xmin,xmax,ymin,ymax)
            newYT, Indice_max = corcetYgraphique(Indicey,YT)
    

            Gmodel = Model(Gauss,independent_vars=['x'])
            params = Gmodel.make_params()
            params['A'].set(50
                        ,vary= True#, min=1, max = 30000000
                        )
            params['u'].set(Indice_max)
            params['sig'].set(50,vary = True, min=0, max = 3000)

            resultGy = Gmodel.fit(newYT, params ,x=Indicey )

            #print(resultGy.fit_report())
            #print('---------------------------------------------------------------------')
            #ymin, yminIndice, ymax, ymaxIndice = optiGy(resultGy.params['u'].value, resultGy.params['sig'].value, Indicex, 3)
            imgZoom = ZoomImage(newimg,xmin,xmax,ymin,ymax)
            #Indiceysig, YTsig = recalcul(Indicey, newYT, yminIndice, ymaxIndice)
            newXT, newIndicex = miseXgraph(newimg,xmin,xmax,ymin,ymax)

            Gmodel = Model(Gauss,independent_vars=['x'])
            params = Gmodel.make_params()
            params['A'].set(50
                        ,vary= True#, min=1, max = 30000000
                        )
            params['u'].set(resultGx.params['u'].value)
            params['sig'].set(50,vary = True, min=0, max = 3000)

            resultGxnew = Gmodel.fit(newXT, params ,x=newIndicex )

            #print(resultGxnew.fit_report())
            #print('---------------------------------------------------------------------')
    
            
            fig, ax = plt.subplots(num = 1, figsize=(9,6),dpi=200)

            plt.plot(Indicex, XT,'ko')
            plt.plot(Indicex, resultGx.best_fit,'r-',label='fit ampl = %.1f , $\mu$ = %.2f , $\sigma$ = %.2f'%(resultGx.params['A'].value,resultGx.params['u'].value,resultGx.params['sig'].value),ms = 1.5)

            ax.xaxis.set_major_locator(AutoLocator())
            ax.xaxis.set_major_formatter('{x:.0f}')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_formatter('{x:.0f}')
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            plt.xlabel(r'X [pixel]',fontsize = 15)
            plt.ylabel(r"Moyenne d'intensité",fontsize = 15)
            plt.grid(True,'major','both',lw = 1.5)
            plt.grid(True,'minor','both',lw = 0.8)

            plt.legend(loc=0, fontsize = 9)
            plt.show()


            fig, ax = plt.subplots(num = 1, figsize=(9,6),dpi=200)
    
            #plt.plot(YT,Indicey,'ko')
            plt.plot(newYT,Indicey,'ko')
            plt.plot(resultGy.best_fit, Indicey,'r-',label='fit ampl = %.1f , $\mu$ = %.2f , $\sigma$ = %.2f'%(resultGy.params['A'].value,resultGy.params['u'].value,resultGy.params['sig'].value),ms = 1.5)
    
    
            #plt.plot(newYT,Indicey,'ko')
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(AutoLocator())
            ax.xaxis.set_major_formatter('{x:.0f}')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_formatter('{x:.0f}')
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            plt.xlabel(r"Moyenne d'intensité",fontsize = 15)
            plt.ylabel(r"Y [pixel]",fontsize = 15)
            plt.grid(True,'major','both',lw = 1.5)
            plt.grid(True,'minor','both',lw = 0.8)

            plt.legend(loc=0, fontsize = 9)
            plt.show()

            fig, ax = plt.subplots(num = 1, figsize=(9,6),dpi=200)

            plt.plot(newIndicex, newXT,'ko')
            plt.plot(newIndicex, resultGxnew.best_fit,'r-',label='fit ampl = %.1f , $\mu$ = %.2f , $\sigma$ = %.2f'%(resultGxnew.params['A'].value,resultGxnew.params['u'].value,resultGxnew.params['sig'].value),ms = 1.5)

            ax.xaxis.set_major_locator(AutoLocator())
            ax.xaxis.set_major_formatter('{x:.0f}')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_formatter('{x:.0f}')
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            plt.xlabel(r'X [pixel]',fontsize = 15)
            plt.ylabel(r"Moyenne d'intensité",fontsize = 15)
            plt.grid(True,'major','both',lw = 1.5)
            plt.grid(True,'minor','both',lw = 0.8)

            plt.legend(loc=0, fontsize = 9)
            plt.show()
            
            print('tache focal trouvé pour %s avec sigx = %.2f et sigy = %.2f'%(path,resultGxnew.params['sig'].value , resultGy.params['sig'].value))
            
            return resultGxnew.params['sig'].value , resultGy.params['sig'].value
        except:
            print('tache focal non-trouvé pour %s'%(path))
    
    return

namefichier, positionvernier, DO = lectureauto('Valeur_mise_tf')
print(namefichier, positionvernier, DO)

#main('tf_14410_DO8_00001')

TMoysigx = []
TMoysigy = []
convertiseur = 1.732 # conversion 1.732µm pour 1 pixel
corection_sigma = 1 #facteur de 2 de base

for tf in range(len(namefichier)):
    print(tf)
    if tf <= 6:
        SUMsigxtf = 0
        SUMsigytf = 0
        nbrpic = 4
        for root in range(1,nbrpic+1):
            sigxtf, sigytf = main(namefichier[tf]+'%.5d'%(root))
            sigxtf = corection_sigma * sigxtf * convertiseur
            sigytf = corection_sigma * sigytf * convertiseur
            print(sigxtf)
            SUMsigxtf = SUMsigxtf + sigxtf
            SUMsigytf = SUMsigytf + sigytf
        
        Moysigx = SUMsigxtf/nbrpic
        Moysigy = SUMsigytf/nbrpic
        TMoysigx.append(Moysigx)
        TMoysigy.append(Moysigy)
        print('Pour %s weist = %.2f'%(namefichier[tf],Moysigx))
    if tf == 7:
        SUMsigxtf = 0
        SUMsigytf = 0
        nbrpic = 2
        sigxtf, sigytf = main(namefichier[tf]+'%.5d'%(1))
        sigxtf = corection_sigma * sigxtf * convertiseur
        sigytf = corection_sigma * sigytf * convertiseur
        SUMsigxtf = SUMsigxtf + sigxtf
        SUMsigytf = SUMsigytf + sigytf
        sigxtf, sigytf = main(namefichier[tf]+'%.5d'%(4))
        sigxtf = corection_sigma * sigxtf * convertiseur
        sigytf = corection_sigma * sigytf * convertiseur
        SUMsigxtf = SUMsigxtf + sigxtf
        SUMsigytf = SUMsigytf + sigytf
        
        Moysigx = SUMsigxtf/nbrpic
        Moysigy = SUMsigytf/nbrpic
        TMoysigx.append(Moysigx)
        TMoysigy.append(Moysigy)
        print('Pour %s weist = %.2f'%(namefichier[tf],Moysigx))
        
del positionvernier[-1]


Pmodel = Model(Poly,independent_vars=['x'])
params = Pmodel.make_params()
params['a'].set(-50,vary= True)
params['b'].set(100)
params['c'].set(50,vary = True)

resultP = Pmodel.fit(TMoysigx, params ,x=positionvernier )

print(resultP.fit_report())
#print('---------------------------------------------------------------------')

Opticsmodel = Model(Gaussianoptics,independent_vars=['z'])
params = Opticsmodel.make_params()
params['wo'].set(40,vary= True)
params['lamb'].set(0.532,vary= False)
params['u'].set(18000)

resultOptic = Opticsmodel.fit(TMoysigx, params ,z=positionvernier )

print(resultOptic.fit_report())
#print('---------------------------------------------------------------------')



posimin = min(positionvernier)
posimax = max(positionvernier)

Xmodel = np.linspace(int(posimin), int(posimax), int(posimax-posimin)+1)
Ymodel = (resultP.params['a'].value * (Xmodel**2)) + (resultP.params['b'].value * Xmodel) + resultP.params['c'].value
Zr = (np.pi*(resultOptic.params['wo'].value)**2)/resultOptic.params['lamb'].value
Opticmodel = resultOptic.params['wo'].value * (1+((Xmodel-resultOptic.params['u'].value)/Zr)**2)**(1/2)
Incertitude_vernier = 2 #µm
Incertutude_lambda = 0.002 #µm
Incertitude_weist = 4 * resultOptic.params['wo'].stderr + Incertitude_vernier + Incertutude_lambda #µm
OpticmodelIncerP = Opticmodel + Incertitude_weist
OpticmodelIncerM = Opticmodel - Incertitude_weist

#Création du graphique waist/Creation of the waist graphic

fig, ax = plt.subplots(num = 1, figsize=(9,6),dpi=200)

plt.plot(positionvernier, TMoysigx,'ko')
#plt.plot(Xmodel, Ymodel,'b-',label=r'fit a = %.3e $x^{2}$ + b = %.3e x + c = %.3f'%(resultP.params['a'].value,resultP.params['b'].value,resultP.params['c'].value),ms = 1.5)

#plt.plot(positionvernier, resultOptic.best_fit,'g-')
plt.plot(Xmodel, Opticmodel ,'r-',label=r'fit $W_{o}$ = %.2f $\mu$m, $Z_{R}$ = %.3f mm , $Z_{0}$ = %.1f $\mu$m'%(resultOptic.params['wo'].value,Zr/1000,resultOptic.params['u'].value),ms = 1.5)
plt.fill_between(Xmodel, Opticmodel, OpticmodelIncerP, where=Opticmodel<OpticmodelIncerP, color='#ffff00')
plt.fill_between(Xmodel, Opticmodel, OpticmodelIncerM, where=Opticmodel>OpticmodelIncerM, color='#ffff00',label='Ecart-Type = $\pm$ %.2f $\mu$m'%(Incertitude_weist))

ax.xaxis.set_major_locator(AutoLocator())
ax.xaxis.set_major_formatter('{x:.0f}')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_major_locator(AutoLocator())
ax.yaxis.set_major_formatter('{x:.0f}')
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel(r'vernier [$\mu$m]',fontsize = 15)
plt.ylabel(r"Rayon de la tache focale [µm]",fontsize = 15)#Moyenne 2$\sigma$ = Waist des TF [$\mu$m]
plt.grid(True,'major','both',lw = 1.5)
plt.grid(True,'minor','both',lw = 0.8)

plt.legend(loc=0, fontsize = 9)
plt.show()


