import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

# GENERISANJE PODATAKA
N = 500
# Generisanje Klase 1
x11 = 0.5*rnd.randn(N, 1)
x21 = 0.5*rnd.randn(N, 1)
K1 = np.concatenate((x11, x21), axis=1)
D1 = np.ones((N, 1))

# Generisanje klase 2
x12 = 0.5*rnd.randn(N, 1) + 2.5
x22 = 0.5*rnd.randn(N, 1) + 2.5
K2 = np.concatenate((x12, x22), axis=1)
D2 = np.zeros((N, 1))

"""
# Prikaz podataka dve klasa - Primetit da su klase linearno seprabilne
plt.figure()
plt.plot(x11, x21, 'o')
plt.plot(x12, x22, '*')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend(['K1', 'K2'])
plt.show()
"""

# Kreiranje ulaza i zeljenog izlaza Perceptrona 
# Bias ce biti predstavljen kao ulaz koji je uvek jednak jedinici
X = np.concatenate((K1, K2), axis=0)
bias = np.ones((2*N, 1))

X = np.concatenate((X, bias), axis=1)
D = np.concatenate((D1, D2), axis=0)

# Kreiranje i trening perceptrona 
# u pitanju je problem klasifikacije oblika u dve klase, potrebno je 
# kreirati LTU perceptron koji se obucava perceptronskim pravilom obucavanja 

# Definisanje funkcije za racunanje izlaza perceptrona 
def predict(x, w):
    act = np.sum(w*x)
    if act >= 0:
        y = 1
    else:
        y = 0
    
    return y

# definisanje funkcije za racunanje promene pojacanja 
def weight_update(lr, d, y, x):
    return lr * (d-y) * x

# Definisanje parametara potrebnih za obucavanje 
W = 0.1*rnd.randn(3) # pocetne tezine 
n = 10      # maksimalan broj epoha za obucavanje 
lr = 0.01   # konstanta za obucavanje 
Emax = 0.02 # Maksimalna dovoljena greska 

# Obucavanje perceptrona: potrebno je u svakoj epohi proci kroz sve primere ulaznog skupa 
for epoch in range(n):      # sve epohe treniranja 
    error = 0
    
    for k in range(len(D)): # Svaki element ulaznog skupa 
        Ypred = predict(W, X[k, :])
        dW = weight_update(lr, D[k], Ypred, X[k, :])
        W += dW

        error += np.abs((D[k] - Ypred))

    error /= np.shape(X)[0]
    print(">epoch=%d, lrate=%.3f, error=%.3f" % (epoch+1, lr, error))

    #if error<Emax: 
    #    break

# Prikaz granice odlucivanja 
# Granica odlucivanja je prava h(x), koja razdvaja prostor obelezja na dva podprostora 
# od kojih svaki odgovara jendoj klasi. Kako je perceptron linearni klasifikator. 
# h(x) = w1*x1 + w2*x2 + wb = 0
# iz prednoh izraza imamo da je: x2 = (-w1/w2)*x1 - wb/w2

w1 = W[0]
w2 = W[1]
wb = W[2]

x1 = np.linspace(-2, 6)
"""
x2 = -w1*x1/w2 - wb/w2
# Prikaz granice odlučivanja
plt.figure()
plt.plot(x11, x21, 'o')
plt.plot(x12, x22, '*')
plt.plot(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim([-2, 6])
plt.grid()
plt.legend(['K1', 'K2', 'h(x)=0'])
plt.show()

"""
x2 = -w1*x1/w2 - wb/w2

# prikaz granice odlucivanja 
plt.figure()
plt.plot(x11, x21, 'o')
plt.plot(x12, x22, '*')
plt.plot(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim([-2, 6])
plt.grid()
plt.legend(['K1', 'K2', 'h(x)=0'])
plt.show()
