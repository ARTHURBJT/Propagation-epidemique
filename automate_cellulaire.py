# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:38:21 2021

@author: Arthur
"""

from random import random, randint
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Implémentation d'un tri fusion
def fusion(T1,T2):
    T=[]
    while len(T1)!=0 and len(T2)!=0:
        if T1[0]<=T2[0]:
            T.append(T1[0])
            T1 = T1 [1:]
        else:
            T.append(T2[0])
            T2 = T2 [1:]
    return T+T1+T2

def tri_fusion(T):
    if len(T)<=1:
        return T
    else:
        return fusion(tri_fusion(T[:round(len(T)/2)]),tri_fusion(T[round(len(T)/2):]))



# Sain : 0
# Contaminé : 1
# Rétablit : 2
# Mort : 3
# case vide : 4

#Ces entiers étant les positions dans le compteur des différents états



def bernoulli(p):
    return random() < p


#creer une ville de taille n (n**2 habitants), 
# sous la forme d'un tableau de tableau. 
# On rajoute à la fin un compteur qui s'actualisera
# des Sain, Contaminé, Rétablit, Mort et case vide presents dans la ville.
def creer_ville (n):    
    V=[]
    C=[0,0,0,0,0]
    for i in range(n):
        E=[]
        for j in range(n):
            if bernoulli(0.07):
                E.append([4,0])
                C[4]+=1
            else:
                E.append([0,0])
                C[0]+=1
        V.append(E)
    V.append(C)
    return V


#n ville de taille p (p**2 habitants)
def creer_pays(n,p):               
    P=[]
    for i in range(n):
        V=creer_ville(p)
        P.append(V)
    return P



def melange_liste(l):
    E=[]
    r=len(l)
    for i in  range(r):
        k=randint(0,len(l)-1)
        E.append(l[k])
        l=l[:k]+l[k+1:]
    return E


#trajets en bus dans une ville, V:ville, p:portion de gens qui se deplacent entre les dates t et t+1
def trajet_bus(V,p):    
    k = len(V)-1
    nb = round((1-p)*k*k)  #nb de gens qui ne se deplacent pas
    bus = []
    C = [i for i in range(k*k)]
    E = []
    for i in range(nb):
        q = randint(0,k*k-1-i)
        E.append(C[q])        #On fait ceci pour ne pas pouvoir tirer deux fois la même personne
        C = C[:q]+C[q+1:]
    E = tri_fusion(E)        #E contient les positions des gens qui ne se déplacent pas
    
    for i in range(k*k):
        if E == []:
            if V[i//k][i%k][0] != 4 and V[i//k][i%k][0] != 3:
                bus.append(V[i//k][i%k])
                V[i//k][i%k] = [4,0]
        else:
            if i == E[0]:
                E = E[1:]
            else:
                if V[i//k][i%k][0] != 4 and V[i//k][i%k][0] != 3:
                    bus.append(V[i//k][i%k])
                    V[i//k][i%k] = [4,0]

    bus = melange_liste(bus)
    Vides = []
    for i in range(k*k):
        if V[i//k][i%k][0] == 4:
            Vides.append(i)      #E contient maintenant les positions des cases vides, ou pourront venir les gens dans le bus
    while bus != []:
        n = randint(0,len(Vides)-1)
        q = bus[0]
        V[Vides[n]//k][Vides[n]%k] = q
        bus = bus[1:]
        Vides = Vides[:n] + Vides[n+1:]        
    return V
    

#l une liste d'élément, on renvoie le nb d'entre eux qui valent 1
def aux1(l): 
    c=0
    for i in l:
        if i==1:
            c+=1
    return c


#determine combien de proches l'element numero n de la ville sont contaminés 
def est_vulnerable(n,V): 
    k=len(V)-1
    if V[n//k][n%k][0]==0:
        if n==0:
            return aux1([V[0][1][0],V[1][0][0],V[1][1][0]])    
        elif n==k-1:
            return aux1([V[0][k-2][0],V[1][k-2][0],V[1][k-1][0]])
        elif n==k*(k-1):
            return aux1([V[k-2][0][0],V[k-2][1][0],V[k-1][1][0]])
        elif n==k*k-1:
            return aux1([V[k-2][k-2][0],V[k-2][k-1][0],V[k-1][k-2][0]])
        elif n//k==0:
            return aux1([V[0][n-1][0],V[0][n+1][0],V[1][n-1][0],V[1][n][0],V[1][n+1][0]])
        elif n//k==k-1:
            return aux1([V[k-1][n%k-1][0],V[k-1][n%k+1][0],V[k-2][n%k-1][0],V[k-2][n%k][0],V[k-2][n%k+1][0]])       
        elif n%k==0:
            return aux1([V[n//k-1][0][0],V[n//k+1][0][0],V[n//k-1][1][0],V[n//k][1][0],V[n//k+1][1][0]])
        elif n%k==k-1:
            return aux1([V[n//k-1][k-1][0],V[n//k+1][k-1][0],V[n//k-1][k-2][0],V[n//k][k-2][0],V[n//k+1][k-2][0]])
        else:
            return aux1([V[n//k-1][n%k-1][0],V[n//k][n%k-1][0],V[n//k+1][n%k-1][0],V[n//k-1][n%k][0],V[n//k+1][n%k][0],V[n//k-1][n%k+1][0],V[n//k][n%k+1][0],V[n//k+1][n%k+1][0]])
    else:
        return 0


#modifie la ville en contaminant ceux qui doivent l'etre avec une probabilité p, si ils sont vulnerables
def contamination(V,p): 
    k=len(V)-1
    E=[]
    for i in range (k*k):
        b=False
        for j in range(est_vulnerable(i,V)):
            b=(b or bernoulli(p))
        if b:
            E.append(i)
    for i in E:
        V[i//k][i%k]=[1,0]
        V[k][1]+=1
        V[k][0]-=1



def apres_inf(V,d,p3,p4,p5): #devenir d'une personne infectée; d=durée de la maladie; p3=proba de mourrir; p4=proba de devenir immunisé; p5=proba de redevenir saint, il reste la probabilité que l'infecté reste infecté
    k=len(V)-1
    for i in range(k*k):
        if V[i//k][i%k][0]==1:
            V[i//k][i%k][1]+=1
            if V[i//k][i%k][1]>=d:
                if bernoulli(p3):
                    V[i//k][i%k]=[3,0]
                    V[k][1]-=1
                    V[k][3]+=1
                elif bernoulli(p4/(1-p3)):  #proba de devenir immun sachant qu'il ne meurt pas
                    V[i//k][i%k]=[2,0]
                    V[k][1]-=1
                    V[k][2]+=1
                elif bernoulli(p5/(1-p3-p4)):
                    V[i//k][i%k]=[0,0]
                    V[k][1]-=1
                    V[k][0]+=1
                
                

def trajet_train(P,p): #P un pays; p la proportion de gens qui prennent le train entre t et t+1
    train = []
    k = len(P[0]) - 1          # taille de la ville 0, mais toutes les villes ont la même taille
    nb = round(p * k * k)      #nb de gens qui se deplacent de chaque ville
    for i in range(len(P)):
        n = 0
        while n != nb:
            q=randint(0,k * k - 1)
            etat = P[i][q//k][q%k][0]
            
            if etat != 3 and etat != 4:
                train.append(P[i][q//k][q%k])
                P[i][q//k][q%k] = [4,0]
                n += 1 
                P[i][k][etat] -= 1  
    train = melange_liste(train)

    for i in range(len(P)):
        wagon = train[:nb]
        E = []
        for j in range(k*k):
            if P[i][j//k][j%k][0] == 4:
                E.append(j)
        while wagon != []:
            n = randint(0,len(E)-1)
            cellule = wagon[0]
            P[i][E[n]//k][E[n]%k] = cellule
            P[i][k][cellule[0]] += 1         #compteur
            wagon=wagon[1:]
            E=E[:n]+E[n+1:]  
        train=train[nb:]
    return P




#fonction utile à l'affichage.
def traduit(P, n, p):
    P0 = []
    for k in range(n):
        V0 = []
        for i in range(p):
            l0 = []
            for j in range(p):
                l0.append(P[k][i][j][0])
            V0.append(l0)
        P0.append(V0)
    return P0





#P0 le pays à modèliser, n un carré entier le nombre de villes de tailles p.
def images(P0, n, p, noms, etats, c1, c2, j):
    fig, axs = plt.subplots(int(sqrt(n)), int(sqrt(n)),figsize=(20,20))
    for i,ax in enumerate(fig.axes):
        g = ax.imshow(P0[i][:p], cmap='jet')
        ax.set_title(noms[i],fontsize=16)
        ax.axis('off')
    # Légende pour la figure
    couleur = [0,1,2,3,4]
    colors = [ g.cmap(g.norm(value)) for value in couleur]
    patches = [ mpatches.Patch(color=colors[j], label=etats[j] ) for j in range(len(couleur)) ]
    
    if c2 != 0:
        plt.suptitle("Carte des villes pendant un confinement du pays à la date "+str(j),fontsize=26,x=0.5,y=0.1)
    elif c1 != [0 for i in range(n)] :
        E=""
        fst=0
        for i in range(n):
            if c1[i]!=0:
                if fst==0:
                    E=str(i+1)
                    fst = 1
                else:
                    E=E+", "+str(i+1)
        if len(E) == 1:
            plt.suptitle("Carte des villes pendant un confinement de la ville "+E,fontsize=26,x=0.5,y=0.1)
        else:
            plt.suptitle("Carte des villes pendant un confinement des villes "+E,fontsize=26,x=0.5,y=0.1)
    else:
        plt.suptitle("Cartes des villes à la date "+str(j),fontsize=26,x=0.5,y=0.1)
    plt.legend(handles=patches, bbox_to_anchor=(0.90, 0), loc=2, borderaxespad=0.,fontsize=20 )
    plt.savefig('.\carte_simul_pays_v9_date_{}.png'.format(j),dpi=100)
    plt.close()



    
def simulation_maladie_dans_pays(n,p,d,t,p1,p2,p3,p4,p5,p6,p7,p8,noms,f): 
    """"n villes de taille p; d=durée de la maldie; l'expérience a une durée de t jours; 
    p1=proba d'etre contaminé en etant vulnerable; p2=part de la pop qui se deplace en bus dans sa ville; 
    Pour un infecté : p3=proba de mourrir, p4=proba de devenir immunisé, p5=proba de redevenir saint; 
    p6=part de la pop d'une ville qui se déplace en train;
    p7=part de la population contaminée à partir de laquelle couvre feu (moins de déplacement entre ville);
    p8=part de la population contaminée à partir de laquelle confinement (moins de déplacement dans la ville);
    noms : noms des villes du pays dans l'ordre ; f la fréquence de traçage de graphique"""
    P=creer_pays(n,p)
    etats = ["Sain","Infecté","rétablit","Mort","case vide"]
    k=randint(0, p*p-1)
    q=randint(0, n-1)
    P[q][k//p][k%p][0]=1
    P[q][p][1]+=1
    P[q][p][0]-=1
    Nb_0=[[] for i in range(n)]
    Nb_1=[[] for i in range(n)]
    Nb_2=[[] for i in range(n)]
    Nb_3=[[] for i in range(n)]
    c1=[0 for i in range(n)] 
    c2=0
    temps=[i for i in range(t)]
    for j in range(t):
        E=[]
        nbinftot=0
        for i in range(n):
            Nb_0[i].append(P[i][p][0])
            Nb_1[i].append(P[i][p][1])
            Nb_2[i].append(P[i][p][2])
            Nb_3[i].append(P[i][p][3])
            if P[i][p][1]!=0:
                apres_inf(P[i],d,p3,p4,p5)
                if c2!=0:
                    contamination(P[i],p1/10)
                    trajet_bus(P[i],p2/10) #car les gens vont faire leurs courses une fois par semaine
                else:
                    if c1[i]==0:
                        if p8>-1:
                            contamination(P[i],p1)
                            trajet_bus(P[i],p2)
                        else:
                            c1[i]+=1
                            contamination(P[i],p1/10)
                            trajet_bus(P[i],p2/10) #car les gens vont faire leurs courses une fois par semaine
                    elif c1[i]==14:
                        c1[i]=0
                        contamination(P[i],p1)
                        trajet_bus(P[i],p2)
                    else:
                        c1[i]+=1
                        contamination(P[i],p1/10)
                        trajet_bus(P[i],p2/10) #car les gens vont faire leurs courses une fois par semaine
                nbinftot+=P[i][p][1]
        if c2==0 :
            if p7>-1 :
                trajet_train(P,p6)
            else:
                c2+=1
                trajet_train(P,p6/5)  #seuls cas exceptionnel peuvent prendre le train, mais il y a deja peu de gens qui le prennent, donc surement deja des cas exceptionnel
        elif c2==30:
            c2=0
            c1 = [0 for i in range(n)]
            trajet_train(P,p6)
        else:
            c2+=1
            trajet_train(P,p6/5)  #seuls cas exceptionnel peuvent prendre le train, mais il y a deja peu de gens qui le prennent, donc surement deja des cas exceptionnel

        if nbinftot==0:
            print(j)
            temps = temps[:j+1]
            break
        if j % f == 0 and f != 1000:
            P0 = traduit(P, n, p)
            images(P0, n, p, noms, etats, c1, c2, j)

    plt.figure()
    plt.title("Nombre d'infectés, sains, morts, immunisés dans le pays")
    plt.xlabel('temps(jours)')
    plt.ylabel("Nombre de personnes")
    for i in range(n):
        plt.plot(temps,Nb_1[i], label = 'Ville'+str(i+1))
    plt.legend()
    plt.savefig('.\graphe_simul_dans_pays_v10_{}.png'.format(i+1),dpi=100)
    plt.close()
    for i in range(n):
        plt.figure()
        plt.title("Nombre d'infectés, sains, morts, immunisés dans la ville "+str(i+1))
        plt.xlabel('temps(jours)')
        plt.ylabel("Nombre de personnes")
        plt.plot(temps,Nb_0[i], label = 'Sains', color='green')
        plt.plot(temps,Nb_1[i], label = 'Infectés', color='red')
        plt.plot(temps,Nb_2[i], label = 'Immunisés', color='blue')
        plt.plot(temps,Nb_3[i], label = 'Morts', color='black')
        plt.legend()
        plt.savefig('.\graphe_simul_dans_ville_v10_{}.png'.format(i+1),dpi=100)
        plt.close()





n=1       #n villes de taille p
p=100
d=7      #d=durée de la maldie en jours

t=100     #l'expérience a une durée de t jours
p1=0.05  #p1=proba d'etre contaminé en etant vulnerable
p2=0.9     #p2=part de la pop qui se deplace en bus dans sa ville

#Pour un infecté :
p3=0.00   #p3=proba de mourrir
p4=0.00       #p4=proba de devenir immunisé
p5=0.9    #p5=proba de redevenir saint

p6=0.02  #p6=part de la pop d'une ville qui se déplace en train
p7=0.15      #p7=part de la population du pays contaminée à partir de laquelle : moins de déplacement entre ville, confinement 30 jours
p8=0.25       #p8=part de la population d'une ville contaminée à partir de laquelle : moins de déplacement dans cette ville et moins de propagation , confinement de 14 jours

noms=["ville "+str(i+1) for i in range(n)]

f = 1000 #frequence de prise de photo

simulation_maladie_dans_pays(n,p,d,t,p1,p2,p3,p4,p5,p6,p7,p8,noms,f)
