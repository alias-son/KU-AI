import numpy as np
from gurobipy import*
import random
import matplotlib.pyplot as plt
import csv

data =[]

data = np.loadtxt("C:/pythondata/processed.cleveland.data",delimiter=',',dtype=str)
L = len(data)
l = len(data[0])-1

cn = len(np.unique(list(zip(*data))[l]))

x=[]

for i in range(L):
    x.append(i)
    for j in range(l):
        if data[i,j]=='?':
            x.remove(i)

data = data[x]
data=data.astype('float64')

print('----------------Class0 and Others-----------------')


l=len(data[0])-1
smallnum = 1.e-3
np.random.seed(0)
np.random.shuffle(data)
train = data
test = data

train = data[:int((len(data))*.80)]
test = data[int(len(data)*.80+1):]

Atrain = list()
Btrain = list()
Atest = list()
Btest = list()

for i in range(len(train)):
    if train[i,-1] == 0.0 :
        Atrain.append(train[i])
    else :
        Btrain.append(train[i])

Atrain = np.array(Atrain)
Btrain = np.array(Btrain)

lA = len(Atrain)
lB = len(Btrain)

print(lA, lB)

m = Model()

w = []
y = []
z = []

r = m.addVar(vtype=GRB.CONTINUOUS, name='r')

for i in range(lA):
    y.append(m.addVar(vtype=GRB.CONTINUOUS, name="y[{}]".format(i), lb = 0))

for i in range(lB):
    z.append(m.addVar(vtype=GRB.CONTINUOUS, name="z[{}]".format(i), lb = 0))

for i in range(l):
    w.append(m.addVar(vtype=GRB.CONTINUOUS, name="w[{}]".format(i), lb = -100, ub = 100))

m.update()

Y = sum(y[i] for i in range(lA))
Z = sum(z[i] for i in range(lB))

m.setObjective(Y/lA + Z/lB, GRB.MINIMIZE)
m.update()

for i in range(lA):
    m.addConstr(sum(Atrain[i][j]*w[j] for j in range(l)) - r + y[i] >= smallnum,
                name="cty[{}]".format(i))

m.update()

for i in range(lB) :
    m.addConstr(sum(Btrain[i][j]*w[j] for j in range(l)) - r + z[i] <= smallnum,
                name = "ctz[{}]".format(i))

m.update()
m.optimize()

# Start Test

for i in range(len(test)):
    if test[i,-1] == 0.0 :
        Atest.append(test[i])
    else :
        Btest.append(test[i])

Atest = np.array(Atest)
Btest = np.array(Btest)

fail = 0

for i in range(len(Atest)):
    if(sum(Atest[i,j]*w[j].x for j in range(l)) < r.x):
        fail += 1

for i in range(len(Btest)):
    if(sum(Btest[i,j]*w[j].x for j in range(l)) > r.x):
        fail += 1

success_rate = (len(test)-fail)*100/len(test)
print(success_rate)

####################################Class1 and Others#########################################
print('----------------Class1 and Others-----------------')
l=len(data[0])-1
smallnum = 1.e-3
np.random.seed(11)
np.random.shuffle(data)
train = data
test = data

train = data[:int((len(data))*.80)]
test = data[int(len(data)*.80+1):]

Atrain = list()
Btrain = list()
Atest = list()
Btest = list()

for i in range(len(train)):
    if train[i,-1] == 1.0 :
        Atrain.append(train[i])
    else :
        Btrain.append(train[i])

Atrain = np.array(Atrain)
Btrain = np.array(Btrain)

lA = len(Atrain)
lB = len(Btrain)

print(lA, lB)

m = Model()

w = []
y = []
z = []

r = m.addVar(vtype=GRB.CONTINUOUS, name='r')

for i in range(lA):
    y.append(m.addVar(vtype=GRB.CONTINUOUS, name="y[{}]".format(i), lb = 0))

for i in range(lB):
    z.append(m.addVar(vtype=GRB.CONTINUOUS, name="z[{}]".format(i), lb = 0))

for i in range(l):
    w.append(m.addVar(vtype=GRB.CONTINUOUS, name="w[{}]".format(i), lb = -100, ub = 100))

m.update()

Y = sum(y[i] for i in range(lA))
Z = sum(z[i] for i in range(lB))

m.setObjective(Y/lA + Z/lB, GRB.MINIMIZE)
m.update()

for i in range(lA):
    m.addConstr(sum(Atrain[i][j]*w[j] for j in range(l)) - r + y[i] >= smallnum,
                name="cty[{}]".format(i))

m.update()

for i in range(lB) :
    m.addConstr(sum(Btrain[i][j]*w[j] for j in range(l)) - r + z[i] <= smallnum,
                name = "ctz[{}]".format(i))

m.update()
m.optimize()

# Start Test

for i in range(len(test)):
    if test[i,-1] == 1.0 :
        Atest.append(test[i])
    else :
        Btest.append(test[i])

Atest = np.array(Atest)
Btest = np.array(Btest)

fail = 0

for i in range(len(Atest)):
    if(sum(Atest[i,j]*w[j].x for j in range(l)) < r.x):
        fail += 1

for i in range(len(Btest)):
    if(sum(Btest[i,j]*w[j].x for j in range(l)) > r.x):
        fail += 1

success_rate = (len(test)-fail)*100/len(test)
print(success_rate)

####################################Class2 and Others#########################################
print('----------------Class2 and Others-----------------')

l=len(data[0])-1
smallnum = 1.e-3
np.random.seed(13)
np.random.shuffle(data)
train = data
test = data

train = data[:int((len(data))*.80)]
test = data[int(len(data)*.80+1):]

Atrain = list()
Btrain = list()
Atest = list()
Btest = list()

for i in range(len(train)):
    if train[i,-1] == 2.0 :
        Atrain.append(train[i])
    else :
        Btrain.append(train[i])

Atrain = np.array(Atrain)
Btrain = np.array(Btrain)

lA = len(Atrain)
lB = len(Btrain)

print(lA, lB)

m = Model()

w = []
y = []
z = []

r = m.addVar(vtype=GRB.CONTINUOUS, name='r')

for i in range(lA):
    y.append(m.addVar(vtype=GRB.CONTINUOUS, name="y[{}]".format(i), lb = 0))

for i in range(lB):
    z.append(m.addVar(vtype=GRB.CONTINUOUS, name="z[{}]".format(i), lb = 0))

for i in range(l):
    w.append(m.addVar(vtype=GRB.CONTINUOUS, name="w[{}]".format(i), lb = -100, ub = 100))

m.update()

Y = sum(y[i] for i in range(lA))
Z = sum(z[i] for i in range(lB))

m.setObjective(Y/lA + Z/lB, GRB.MINIMIZE)
m.update()

for i in range(lA):
    m.addConstr(sum(Atrain[i][j]*w[j] for j in range(l)) - r + y[i] >= smallnum,
                name="cty[{}]".format(i))

m.update()

for i in range(lB) :
    m.addConstr(sum(Btrain[i][j]*w[j] for j in range(l)) - r + z[i] <= smallnum,
                name = "ctz[{}]".format(i))

m.update()
m.optimize()

# Start Test

for i in range(len(test)):
    if test[i,-1] == 2.0 :
        Atest.append(test[i])
    else :
        Btest.append(test[i])

Atest = np.array(Atest)
Btest = np.array(Btest)

fail = 0

for i in range(len(Atest)):
    if(sum(Atest[i,j]*w[j].x for j in range(l)) < r.x):
        fail += 1

for i in range(len(Btest)):
    if(sum(Btest[i,j]*w[j].x for j in range(l)) > r.x):
        fail += 1

success_rate = (len(test)-fail)*100/len(test)
print(success_rate)

####################################Class3 and Others#########################################
print('----------------Class3 and Others-----------------')

l=len(data[0])-1
smallnum = 1.e-3
np.random.seed(41)
np.random.shuffle(data)
train = data
test = data

train = data[:int((len(data))*.80)]
test = data[int(len(data)*.80+1):]

Atrain = list()
Btrain = list()
Atest = list()
Btest = list()

for i in range(len(train)):
    if train[i,-1] == 3.0 :
        Atrain.append(train[i])
    else :
        Btrain.append(train[i])

Atrain = np.array(Atrain)
Btrain = np.array(Btrain)

lA = len(Atrain)
lB = len(Btrain)

print(lA, lB)

m = Model()

w = []
y = []
z = []

r = m.addVar(vtype=GRB.CONTINUOUS, name='r')

for i in range(lA):
    y.append(m.addVar(vtype=GRB.CONTINUOUS, name="y[{}]".format(i), lb = 0))

for i in range(lB):
    z.append(m.addVar(vtype=GRB.CONTINUOUS, name="z[{}]".format(i), lb = 0))

for i in range(l):
    w.append(m.addVar(vtype=GRB.CONTINUOUS, name="w[{}]".format(i), lb = -100, ub = 100))

m.update()

Y = sum(y[i] for i in range(lA))
Z = sum(z[i] for i in range(lB))

m.setObjective(Y/lA + Z/lB, GRB.MINIMIZE)
m.update()

for i in range(lA):
    m.addConstr(sum(Atrain[i][j]*w[j] for j in range(l)) - r + y[i] >= smallnum,
                name="cty[{}]".format(i))

m.update()

for i in range(lB) :
    m.addConstr(sum(Btrain[i][j]*w[j] for j in range(l)) - r + z[i] <= smallnum,
                name = "ctz[{}]".format(i))

m.update()
m.optimize()

# Start Test

for i in range(len(test)):
    if test[i,-1] == 3.0 :
        Atest.append(test[i])
    else :
        Btest.append(test[i])

Atest = np.array(Atest)
Btest = np.array(Btest)

fail = 0

for i in range(len(Atest)):
    if(sum(Atest[i,j]*w[j].x for j in range(l)) < r.x):
        fail += 1

for i in range(len(Btest)):
    if(sum(Btest[i,j]*w[j].x for j in range(l)) > r.x):
        fail += 1

success_rate = (len(test)-fail)*100/len(test)
print(success_rate)

####################################Class4 and Others#########################################
print('----------------Class4 and Others-----------------')

l=len(data[0])-1
smallnum = 1.e-3
np.random.seed(11)
np.random.shuffle(data)
train = data
test = data

train = data[:int((len(data))*.80)]
test = data[int(len(data)*.80+1):]

Atrain = list()
Btrain = list()
Atest = list()
Btest = list()

for i in range(len(train)):
    if train[i,-1] == 4.0 :
        Atrain.append(train[i])
    else :
        Btrain.append(train[i])

Atrain = np.array(Atrain)
Btrain = np.array(Btrain)

lA = len(Atrain)
lB = len(Btrain)

print(lA, lB)

m = Model()

w = []
y = []
z = []

r = m.addVar(vtype=GRB.CONTINUOUS, name='r')

for i in range(lA):
    y.append(m.addVar(vtype=GRB.CONTINUOUS, name="y[{}]".format(i), lb = 0))

for i in range(lB):
    z.append(m.addVar(vtype=GRB.CONTINUOUS, name="z[{}]".format(i), lb = 0))

for i in range(l):
    w.append(m.addVar(vtype=GRB.CONTINUOUS, name="w[{}]".format(i), lb = -100, ub = 100))

m.update()

Y = sum(y[i] for i in range(lA))
Z = sum(z[i] for i in range(lB))

m.setObjective(Y/lA + Z/lB, GRB.MINIMIZE)
m.update()

for i in range(lA):
    m.addConstr(sum(Atrain[i][j]*w[j] for j in range(l)) - r + y[i] >= smallnum,
                name="cty[{}]".format(i))

m.update()

for i in range(lB) :
    m.addConstr(sum(Btrain[i][j]*w[j] for j in range(l)) - r + z[i] <= smallnum,
                name = "ctz[{}]".format(i))

m.update()
m.optimize()

# Start Test

for i in range(len(test)):
    if test[i,-1] == 4.0 :
        Atest.append(test[i])
    else :
        Btest.append(test[i])

Atest = np.array(Atest)
Btest = np.array(Btest)

fail = 0

for i in range(len(Atest)):
    if(sum(Atest[i,j]*w[j].x for j in range(l)) < r.x):
        fail += 1

for i in range(len(Btest)):
    if(sum(Btest[i,j]*w[j].x for j in range(l)) > r.x):
        fail += 1

success_rate = (len(test)-fail)*100/len(test)
print(success_rate)
