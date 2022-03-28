# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:38:21 2021
@author: Arthur
"""
from random import random, randint
from math import *
import matplotlib.pyplot as plt


# Implémentation d'un tri fusion
def fusion(T1, T2):
    T = []
    while len(T1) != 0 and len(T2) != 0:
        if T1[0] <= T2[0]:
            T.append(T1[0])
            T1 = T1 [1:]
        else:
            T.append(T2[0])
            T2 = T2 [1:]
    return T + T1 + T2

def tri_fusion(T):
    if len(T) <= 1:
        return T
    else:
        return fusion(tri_fusion(T[ : round(len(T) / 2)]), tri_fusion(T[round(len(T) / 2) : ]))


def bernoulli(p):
    return random() < p

# Creer une ville de taille p (p**2 habitants), 
# sous la forme d'une matrice. 
# On rajoute à la fin un compteur qui s'actualisera
# des Sain, Contaminé, Rétablit, Mort et case vide presents dans la ville.
def creer_ville (p):
    nb_vide = round(p * p * 0.07) #le nombre de case qui seront vides
    C=[p * p - nb_vide,0,0,0,nb_vide]
    V = [[[0,0] for i in range(p)] for i in range(p)]
    entiers = [i for i in range(p * p)]

    for k in range(nb_vide):
        q = randint(0,len(entiers) - 1 - k)
        i = entiers[q]
        V[i//p][i%p] = [4,0]
        entiers = entiers[:q]+entiers[q+1:] #ainsi on ne peut tirer deux fois la même case

    V.append(C)
    return V

       
#n villes, la ville i est de taille lp[i] (lp[i]**2 habitants).
def creer_pays(n,lp):               
    P=[]
    for i in range(n):
        V = creer_ville(lp[i])
        P.append(V)
    return P


def melange_liste(l):
    E = []
    r = len(l)
    for i in  range(r):
        k = randint(0,r - 1 - i)
        E.append(l[k])
        l = l[:k] + l[k+1:]
    return E


# trajets en bus dans une ville, V : ville, 
# p : portion de gens qui se deplacent quotidiennement dans cette ville.
def trajet_bus(V,p):    
    k = len(V) - 1
    nb = round((1 - p) * k * k)  #nb de gens qui ne se deplacent pas
    bus = []
    entiers = [i for i in range(k * k)]
    E = []
    for i in range(nb):
        q = randint(0,k * k - 1 - i)
        E.append(entiers[q])
        entiers = entiers[:q] + entiers[q+1:]
    E = tri_fusion(E)        #E contient les positions des gens qui ne se déplacent pas
    if E == []:
        for i in range(k * k):
            if V[i//k][i%k][0] != 4 and V[i//k][i%k][0] != 3:
                bus.append(V[i//k][i%k])     #montée dans le bus
                V[i//k][i%k] = [4,0]         #la case devient donc vide
    else:
        #tout le monde monte sauf les personnes tirée au sort avant (dans E) et les morts
        for i in range(k * k):
            if i == E[0]:
                if len(E) > 1:             
                    E = E[1:]
            else:
                if V[i//k][i%k][0] != 4 and V[i//k][i%k][0] != 3:
                    bus.append(V[i//k][i%k])
                    V[i//k][i%k] = [4,0]
    bus = melange_liste(bus)
    Case_vide = []
    for i in range(k * k):
        if V[i//k][i%k][0] == 4:
            Case_vide.append(i)      #Case_vide contient les positions des cases vides
    nb_vides = len(Case_vide)
    long_bus = len(bus)
    for j in range(long_bus):
        n = randint(0, nb_vides - 1 - j)
        q = randint(0,long_bus - 1 - j)
        i = Case_vide[n]
        V[i//k][i%k] = bus[q]        #descente du bus dans une case vide
        bus = bus[:q] + bus[q+1:]
        Case_vide = Case_vide[:n] + Case_vide[n+1:]        
    return V
    

# l une liste d'élément, la fonction renvoie k
# si il y a k fois la valeur 1 dans l
def aux1(l): 
    c = 0
    for i in l:
        if i == 1:
            c += 1
    return c


# la fonction determine le nombre de voisins
# de l'element en place n de la ville V sont infectés.
def est_vulnerable(n,V): 
    k = len(V) - 1
    if V[n//k][n%k][0] == 0: #on ne s'intéresse qu'aux personnes encore saines
        if n == 0:             #coin supérieur gauche
            return aux1([V[0][1][0],V[1][0][0],V[1][1][0]])    
        elif n == k - 1:       #coin supérieur droit
            return aux1([V[0][k-2][0],V[1][k-2][0],V[1][k-1][0]])
        elif n == k * (k - 1): #coin inférieur gauche
            return aux1([V[k-2][0][0],V[k-2][1][0],V[k-1][1][0]])
        elif n == k * k - 1:   #coin inférieur droit 
            return aux1([V[k-2][k-2][0],V[k-2][k-1][0],V[k-1][k-2][0]])
        elif n//k == 0:        #arrete du haut
            return aux1([V[0][n-1][0],V[0][n+1][0],V[1][n-1][0],V[1][n][0],V[1][n+1][0]])
        elif n//k == k - 1:    #arrete du bas
            return aux1([V[k-1][n%k-1][0],V[k-1][n%k+1][0],V[k-2][n%k-1][0],V[k-2][n%k][0],V[k-2][n%k+1][0]])       
        elif n%k == 0:         #arrete de gauche
            return aux1([V[n//k-1][0][0],V[n//k+1][0][0],V[n//k-1][1][0],V[n//k][1][0],V[n//k+1][1][0]])
        elif n%k == k-1:       #arrete de droite
            return aux1([V[n//k-1][k-1][0],V[n//k+1][k-1][0],V[n//k-1][k-2][0],V[n//k][k-2][0],V[n//k+1][k-2][0]])
        else:                  #centre
            return aux1([V[n//k-1][n%k-1][0],V[n//k][n%k-1][0],V[n//k+1][n%k-1][0],V[n//k-1][n%k][0],V[n//k+1][n%k][0],V[n//k-1][n%k+1][0],V[n//k][n%k+1][0],V[n//k+1][n%k+1][0]])
    else:
        return 0


#modifie la ville en contaminant les personnes vulnérables avec une probabilité p.
def contamination(V,p): 
    k = len(V) - 1
    E = []
    for i in range (k * k):
        b = False
        #si une personne a n voisins contaminés, elle a n fois plus de chance de le devenir
        for j in range(est_vulnerable(i,V)): 
            b = (b or bernoulli(p))
        if b:
            E.append(i)
    for i in E:
        V[i//k][i%k] = [1,0]
        V[k][1] += 1
        V[k][0] -= 1


# Devenir d'une personne infectée; d = durée de la maladie; p3=probabilité de mourrir; 
# p5=probabilité de redevenir saint, il reste la probabilité que l'infecté reste infecté
def apres_inf(V,d,p3,p5): 
    k = len(V) - 1
    for i in range(k * k):
        # partie infecté :
        if V[i//k][i%k][0] == 1:
#On passe chaque jour dans apres_inf, on en profite donc pour ajouter 1 au compteur de durée de maladie des infectés
            V[i//k][i%k][1] += 1    
            if V[i//k][i%k][1] >= d:
                #passé la durée de la maldie, la malade peut évoluer
                if bernoulli(p3):
                    V[i//k][i%k] = [3,0]
                    V[k][1] -= 1
                    V[k][3] += 1
                elif bernoulli(p5/(1-p3)):
                    V[i//k][i%k] = [0,0]
                    V[k][1] -= 1
                    V[k][0] += 1
                # sinon, il reste infecté
        # partie vacciné :
        if V[i//k][i%k][0] == 2:
#On passe chaque jour dans apres_inf, on en profite donc pour ajouter 1 au compteur de durée de vaccin des vaccinés
            V[i//k][i%k][1] += 1
            if V[i//k][i%k][1] >= 50:  #le vaccin est supposé efficace 50 jours
                V[i//k][i%k] = [0,0]   # la personne redevient saine
                V[k][2] -= 1
                V[k][0] += 1


# p la portion de gens qui se déplacent en train (àl'échelle nationale) ; 
# P un pays ; pop sont nombre d'habitants.
# La fonction renvoie la matrice de déplacement associée à cette situation.
def creer_mat_dep(p,P,pop): 
    n = len(P)
    m = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        h = (len(P[i]) - 1) ** 2 # nb d'habitants de la ville i
        for j in range(n):
            if i != j:
                m[i][j] = round((len(P[j]) - 1) ** 2 * h * p / pop)
    return m


#P un pays; mat_dep la matrice de déplacement
def trajet_train(P,mat_dep): 
    n = len(P)
    gare = []   # va accueuillir les trains, puis les renvoyer dans les bonnes villes
    for i in range(n):
        train = []     # va accueuillir tout les voyageur allant à la ville i
        for j in range(n):
            wagon = []    # le wagon allant de la ville j à la ville i
            if i != j:
                p = len(P[j]) -1 #population de la ville j
                nb = mat_dep[j][i]  #nb de gens qui se deplacent
                while nb != 0:
                    q=randint(0,p*p-1)                    
                    etat = P[j][q//p][q%p][0]
                    if etat != 3 or etat != 4:
                        wagon.append(P[j][q//p][q%p])
                        P[j][q//p][q%p] = [4,0]
                        nb -= 1 
                        P[j][p][etat] -= 1  #on ajoute pas de case vide car on va la remplir juste après
            wagon = melange_liste(wagon)
            train += wagon
        gare.append(train)    # ainsi on maitrise les destinations, le train i va à la ville i.

    for i in range(n): #redistribution
        train = gare[i] #on reparti les personnes du train numéro i (ceux allant à i) dans la ville i
        p = len(P[i]) - 1
        case_vide = []
        for j in range(p * p):
            if P[i][j//p][j%p][0] == 4: #on regarde les indices des cases vides pour distribuer dans celle-ci
                case_vide.append(j)
        long = len(case_vide)
        for k in range(len(train)):
            r = randint(0,long - 1 - k)
            c = case_vide[r]
            P[i][c//p][c%p] = train[k]
            etat = train[k][0]
            P[i][p][etat] += 1
            case_vide = case_vide[:r] + case_vide[r+1:] 
    return P


#P un pays de n ville, on va vacciner nb_vax personnes dans une des villes
def vaccination(nb_vax,P,n):
    q = randint(0, n-1)
    tailleq = len(P[q]) - 1
    vaccinables = []
    for k in range(tailleq * tailleq):
        etat = P[q][k//tailleq][k%tailleq][0]
        if etat == 0 or etat == 1:
            vaccinables.append(k)
    #on ne peut pas vacciner plus de gens que le nombre de gens vaccinables (personnes saines ou infectés)
    for i in range(min(nb_vax,len(vaccinables))): 
        r = randint(0,len(vaccinables) -1)
        etat = P[q][r//tailleq][r%tailleq][0]
        P[q][r//tailleq][r%tailleq] = [2,1]
        P[q][tailleq][etat] -= 1
        P[q][tailleq][2] += 1


#fonction utile à l'affichage.
def traduit(P, n):
    P0 = []
    for k in range(n):
        p = len(P[k]) - 1
        V0 = []
        for i in range(p):
            l0 = []
            for j in range(p):
                l0.append(P[k][i][j][0]) 
#P0 est le pays P où les habitants ne sont plus un couple d'informations, mais juste une seule : leur état.
            V0.append(l0)
        P0.append(V0)
    return P0


#P0 le pays à modèliser (sous forme traduite), n un carré d'entier : le nombre de villes.
def images(P0, n, noms, etats, c1, c2, j):
    fig, axs = plt.subplots(int(sqrt(n)), int(sqrt(n)),figsize=(20,20))
    for i,ax in enumerate(fig.axes):
        p = len(P0[i])
        g = ax.imshow(P0[i], cmap = 'jet')   #affichage de la ville i
        ax.set_title(noms[i],fontsize = 16)
        ax.axis('off')
    # Légende pour la figure
    couleur = [0,1,2,3,4]
    colors = [ g.cmap(g.norm(value)) for value in couleur]
    patches = [ mpatches.Patch(color = colors[j], label = etats[j] ) for j in range(len(couleur)) ]
    #on informe d'un confinement en cours dans le titre de l'image :
    if c2 != 0:   
        plt.suptitle("Carte des villes pendant un confinement du pays à la date " + str(j),fontsize=26,x=0.5,y=0.1)
    elif c1 != [0 for i in range(n)] :
        E = ""
        fst = 0
        for i in range(n): #on selectionne les villes confinées
            if c1[i] != 0:
                if fst == 0:
                    E = str(i+1)
                    fst = 1
                else:
                    E = E + ", " + str(i + 1)
        if len(E) == 1:
            plt.suptitle("Carte des villes pendant un confinement de la ville "+E,fontsize=26,x=0.5,y=0.1)
        else:
            plt.suptitle("Carte des villes pendant un confinement des villes "+E,fontsize=26,x=0.5,y=0.1)
    else:
        plt.suptitle("Cartes des villes à la date "+str(j),fontsize=26,x=0.5,y=0.1)   #on informe la date
    plt.legend(handles = patches, bbox_to_anchor = (0.90, 0), loc = 2, borderaxespad = 0.,fontsize = 20 )
    plt.savefig('.\carte_simul_v16_date_{}.png'.format(j),dpi = 100)
    plt.close()


# matrice est un matrice carrée, scal est un scalaire.
# la fonction renvoie 1/scal * matrice
def div(matrice,scal): 
    n = len(matrice)
    m = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            m[i][j] = matrice[i][j] * scal
    return m


def simulation_maladie_dans_pays(n,lp,d,t,p1,p2,p3,fct_vax,p5,p6,p7,p8,noms,f): 
    """"n villes, leurs tailles sont dans lp; d=durée de la maladie (jours); l'expérience a une durée de t jours; 
    p1=proba d'être contaminé en étant vulnerable; p2=part de la pop des villes qui se deplace en bus; 
    p3=proba de mourrir pour un infecté; fct_vax : rythme de vaccination; 
    p5=proba de redevenir saint après une infection; p6=part de la pop du pays qui se déplace en train;
    p7=part de la population du pays contaminée à partir de laquelle confinement nationnale;
    p8=part de la population d'une ville contaminée à partir de laquelle confinement de cette ville"""
    P = creer_pays(n,lp)
    pop = 0
    for i in range(len(lp)):
        pop += lp[i] ** 2     #pop sera le nombre d'habitant du pays
    mat_dep = creer_mat_dep(p6,P,pop)
    etats = ["Sain","Infecté","rétablit","Mort","case vide"] #dans l'ordre (à la place i il y a l'état n°i)
    #contamination d'une personne :
    q = randint(0, n - 1)
    tailleq = len(P[q]) - 1
    while P[q][tailleq][1] != 1: #on attend qu'il y aie effectivement un infecté
        k = randint(0,tailleq * tailleq -1)
        if P[q][k//tailleq][k%tailleq][0] == 0:
            P[q][k//tailleq][k%tailleq][0] = 1
            P[q][tailleq][1] += 1
            P[q][tailleq][0] -= 1
    Nb_0 = [[] for i in range(n)]  #Nb_0[i][t] contiendra le nombre de sains (etat = 0) de la ville i à la date t
    Nb_1 = [[] for i in range(n)]  #Nb_0[i][t] contiendra le nombre de infectés (etat = 1) de la ville i à la date t
    Nb_2 = [[] for i in range(n)]  #Nb_0[i][t] contiendra le nombre de vaccinés (etat = 2) de la ville i à la date t
    Nb_3 = [[] for i in range(n)]  #Nb_0[i][t] contiendra le nombre de morts (etat = 3) de la ville i à la date t
    c1 = [0 for i in range(n)]  
    #si c1[i] = 0, la ville i n'est pas confinée, sinon c1[i]=t, la ville i est confinée depuis t jours
    c2 = 0
    #si c2 = 0, le pays n'est pas confinée, sinon c2=t, le pays est confinée depuis t jours
    temps = [i for i in range(t)]
    for j in range(t):
        if j >= 100:  #mise en place de la vaccination
            vaccination(round(fct_vax(j - 100)),P,n)
            #la fonction fct_vax nous donne le nombre de personne qui seront vaccinés à la date j
        E = []
        nbinftot = 0
        for i in range(n):
            taillei = len(P[i]) - 1
            Nb_0[i].append(P[i][taillei][0])
            Nb_1[i].append(P[i][taillei][1])
            Nb_2[i].append(P[i][taillei][2])
            Nb_3[i].append(P[i][taillei][3])
            if P[i][taillei][1] != 0:   #on fait évoluer la ville seulement si elle contient des infectés
                #les infectés qui le sont depuis plus de 7j évolue, deviennent mort, sain ou restent infectés
                #les vaccinés qui le sont depuis plus de 50j redeviennent sain
                apres_inf(P[i],d,p3,p5)  
                if c2 != 0: #critère de confinement nationnal
                    contamination(P[i],p1/10) #on reduit de 10 les trajets dans les villes, et les contaminations
                    trajet_bus(P[i],p2/10) 
                else:
                    if c1[i] == 0: #la ville i n'est pas confinée
                        if p8>(P[i][taillei][1]/(taillei*taillei)):
                            contamination(P[i],p1)
                            trajet_bus(P[i],p2)
                        else: #on a dépassé le seuil, le confinement de cette ville commence
                            c1[i] += 1
                            contamination(P[i],p1/10)
                            trajet_bus(P[i],p2/10) 
                    elif c1[i]==14: #on est arrivé à la fin du confinement de la ville i
                        c1[i]=0
                        contamination(P[i],p1)
                        trajet_bus(P[i],p2)
                    else:   #on est en court de confinement de la ville i
                        c1[i]+=1
                        contamination(P[i],p1/10)
                        trajet_bus(P[i],p2/10) 
                nbinftot+=P[i][taillei][1] 
                #cette contiendra le nombre d'infecté du pays à la date j à la fin de cette boucle
        if c2==0 : #il n'y a pas de confinement nationnal en court
            if p7>(nbinftot/(pop)) :
                trajet_train(P, mat_dep)
                print(1)
            else: #on a dépassé le seuil, on déclanche le confinement nationnal
                c2+=1
                trajet_train(P, div(mat_dep,5))  #on reduit de 5 les déplacements entre villes
        elif c2==30: #fin du confinement nationnal
            c2=0
            c1 = [0 for i in range(n)]
            trajet_train(P, mat_dep)
        else: #on est en court de confinement nationnal
            c2+=1
            trajet_train(P, div(mat_dep,5))  

        if nbinftot==0: #la maladie s'est éteinte
            print(j)  #le programme nous informe de la date de l'héradication de la maladie
            temps = temps[:j+1]
            break
        #on produit une image tout les k*f jours, k un entier
        #1000 est le code indiquant qu'on ne veut pas produire d'image (fait gagner beaucoup en temps de calculs)
        if j % f == 0 and f != 1000: 
            P0 = traduit(P, n)
            images(P0, n, noms, etats, c1, c2, j)

    plt.figure() #on produit le graphe du nombre d'infectés dans chaque ville
    plt.title("Nombre d'infectés dans les villes")
    plt.xlabel('temps(jours)')
    plt.ylabel("Nombre de personnes")
    for i in range(n):
        plt.plot(temps,Nb_1[i], label = 'Ville'+str(i+1))
    plt.legend()
    plt.savefig('.\graphe_simul_v20_{}.png'.format(1),dpi=100)
    plt.close()

    #on calcul la liste des nombres de sains, infectés, vaccinés, morts aux dates j à l'échelle nationnale
    #en sommant les listes que l'on avait pour chaque villes
    Nb_0_tot=[]
    Nb_1_tot=[]
    Nb_2_tot=[]
    Nb_3_tot=[]
    for j in temps:
        tot0 = 0
        tot1 = 0
        tot2 = 0
        tot3 = 0
        for i in range(n):
            tot0 += Nb_0[i][j]
            tot1 += Nb_1[i][j]
            tot2 += Nb_2[i][j]
            tot3 += Nb_3[i][j]
        Nb_0_tot.append(tot0)
        Nb_1_tot.append(tot1)
        Nb_2_tot.append(tot2)
        Nb_3_tot.append(tot3)
    plt.figure() #on produit le graghe des nombres de sain, infecté, vacciné, mort à l'échelle nationnale
    plt.title("Nombre d'infectés, sains, morts, immunisés dans le pays ")
    plt.xlabel('temps(jours)')
    plt.ylabel("Nombre de personnes")
    plt.plot(temps,Nb_0_tot, color='green', label = 'Sains')
    plt.plot(temps,Nb_1_tot, color='red', label = 'Infectés')
    plt.plot(temps,Nb_2_tot, color='blue', label = 'Vaccinés')
    plt.plot(temps,Nb_3_tot, color='black', label = 'Morts')
    plt.legend()
    plt.savefig('.\graphe_simul_dans_pays_v20_{}.png'.format(1),dpi=100)
    plt.close()


n=9        #n villes, n doit être un carré d'entier naturel
lp=[84,130,93,146,129,89,95,113,98] #les tailles des villes, lp[i] est la taille de la ville i
d=7        #d=durée de la maldie en jours

t=3     #l'expérience a une durée de t jours
p1=0.07    #p1=proba d'etre contaminé en etant vulnerable
p2=0.9     #p2=part de la pop qui se deplace en bus tout les jours

p3=0.0061  #p3=proba de mourrir pour un infecté

def fct_vax(t):
    f = t ** 4 * cos(t)
    return (abs(f) + f) /2 
           #rythme de vaccination une fois qu'elle est déclanchée (100 jours)

p5=0.9     #p5=proba de redevenir saint

p6=0.01    #p6 = part de la pop d'une ville qui se déplace en train
p7=0.20    #p7 = taux de contamination du pays à partir duquel : 
           #  moins de déplacement entre ville, 
           #  et confinement de toute les villes pendant 30 jours
p8=0.25    #p8 = taux de contamination du pays à partir duquel : 
           #  moins de déplacement dans cette ville, 
           #  moins de propagation pendant 14 jours

noms=["ville "+str(i+1) for i in range(n)]
f = 100

simulation_maladie_dans_pays(n,lp,d,t,p1,p2,p3,fct_vax,p5,p6,p7,p8,noms,f)


[[0, 11, 6,  14, 11, 5,  6,  8,   6], 
[11, 0,  13, 33, 26, 12, 14, 20, 15], 
[6,  13, 0,  17, 13, 6,  7,  10,  8],
[14, 33, 17, 0,  32, 15, 17, 25, 19], 
[11, 26, 13, 32, 0,  12, 14, 19, 15], 
[5,  12, 6,  15, 12, 0,  7,  9,   7], 
[6,  14, 7,  18, 14, 7,  0,  10,  8], 
[8,  20, 10, 25, 19, 9,  10, 0,  11], 
[6,  15, 8,  19, 15, 7,  8,  11, 0]]