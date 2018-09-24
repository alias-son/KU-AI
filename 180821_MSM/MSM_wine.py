import numpy as np
from gurobipy import *
import random
data=[]

data = np.loadtxt("C:/pythondata/wine.data",delimiter=',',dtype=str)

L=len(data)
l=len(data[0])-1

# Remove ?
x=[]
for i in range(L):
    x.append(i)
    for j in range(l):
        if data[i,j]=='?':
            x.remove(i)
data = data[x]
data = data.astype('float64')

L=len(data)
l = len(data[0])-1
data1 = np.c_[data,np.zeros(L)] # Add a column as zero value to the right of the data (for checking classification)

np.random.shuffle(data1)

train=data1[:int((len(data))*.80)]
test=data1[int(len(data)*.80):]

ci =1 # class identifier
smallnum=1.e-3
bignum=1.e+3
ltr=len(train)

# Variable declaration
Atrain=list()
Btrain=list()
apl=list()
bpl=list()
Cl=list()

# Divide train set by A and B
for i in range(ltr):
    if train[i, 0] == ci:
        Atrain.append(train[i])
    else :
        Btrain.append(train[i])

Atrain = np.array(Atrain)
Btrain = np.array(Btrain)


pl=[]
rsl = []
itera = 1
oldfuzzy = []

# Coding to find a random number whose pp value is greater than 0.5 <- using random seed
while itera > 0:
    p = [0]*l
    rs = random.random() # setting result value by fixing seed
    random.seed(rs)
    rsl.append(rs) # compare whether result value is same after saving random seed

    # Continue to find random values as an infinite loop
    while 0 == 0:
        for i in range(l):
            p[i] += random.uniform(-1.0, 1.0)
        if sum(p[i]*p[i] for i in range(l)) >= 0.5:  # if pp value is greater than 0.5, end
            pl.append(p)
            break

    lA = len(Atrain)
    lB = len(Btrain)

    # Alternative Linear Separability theory, method2 iteration
    while 1 == 1:
        m = Model()
        # alpha, beta declaration
        ap = m.addVar(vtype=GRB.CONTINUOUS, name='ap', ub=bignum, lb=-bignum)
        bp = m.addVar(vtype=GRB.CONTINUOUS, name='bp', ub=bignum, lb=-bignum)

        # Objective Declaration
        m.setObjective(ap - bp, GRB.MAXIMIZE)


        C = []
        for i in range(l):
            C.append(m.addVar(vtype=GRB.CONTINUOUS, name = "C[{}]".format(i), ub=bignum, lb=-bignum))

        # Constraint condition Coding
        for i in range(lA):
            if Atrain[i, -1] == 0:
                m.addConstr(sum(C[j]*Atrain[i,j] for j in range(l)) - ap - smallnum >= 0, name = "ctp[{}]".format(i))

        for i in range(lB):
            if Btrain[i, -1] == 0:
                m.addConstr(sum(C[j]*Btrain[i,j] for j in range(l)) - bp + smallnum <= 0, name = "ctq[{}]".format(i))

        m.addConstr(sum(p[i]*C[i] for i in range(l)) >= .5*(.5+sum(p[i]*p[i] for i in range(l))), name = "null")

        m.update()
        m.optimize()

        # if value is convergence to C, to break
        k = 0
        for i in range(l):
            if (abs(C[i].x - p[i]) < smallnum):
                k += 1

        if k == l:
            break

        # If it does not converge, put p into c and then perform 'while' again
        for i in range(l):
            p[i] = C[i].x


###############################################################################

# Check if C value is the same as previous C
    CI= 0

    for i in range(l):
        if itera >1 and abs(
                [-1][i] - C[i].x)>smallnum:
            CI+=1

    Cll = []


# The first iteration
    if itera == 1:
        for i in range(l):
            Cll.append(C[i].x)

        Cl.append(Cll)
        apl.append(ap.x)
        bpl.append(bp.x)

        sA, sB = 0, 0        # Variable declaration for counting number of successful points

        for i in range(lA):
            if (Atrain[i, -1] == 0)  and (sum(C[j].x * Atrain[i, j] for j in range(l)) - bp.x > 0):
                Atrain[i, -1] =1        # If successful, replace the last field with a value of 1
                sA +=1
        print("Number of classifeid A points is : ", sA)

        for i in range(lB):
            if (Btrain[i, -1] == 0)  and (sum(C[j].x * Btrain[i, j] for j in range(l)) - ap.x < 0):
                Btrain[i, -1] =1
                sB +=1
        print("Number of classifeid B points is : ", sB)

        #print(Atrain)
        #print(Btrain)


        # The number of points that are mixed is not recognized and classified, and the last field value is zero

        fuzzy = 0
        for i in range(lA):
            if Atrain[i, -1] == 0:
                fuzzy += 1

        for i in range(lB):
            if Btrain[i, -1] == 0:
                fuzzy += 1

        if ap.x >= bp.x or fuzzy == 0:
            itera = 0
            print('-----')
            print('Successful termination with 0 misclassification!')
        else:
            oldfuzzy.append(fuzzy)
            itera += 1

# The second iteration
    elif itera>1:

        if CI ==l and abs(apl[-1] -ap.x) > smallnum and abs(bpl[-1] -bp.x) >smallnum:

            for i in range(l):
                Cll.append(C[i].x)

            Cl.append(Cll)
            apl.append(ap.x)
            bpl.append(bp.x)

            sA,sB = 0,0

            for i in range(lA):
                if(Atrain[i][-1]==0) and (sum(C[j].x*Atrain[i,j] for j in range(l)) -bp.x > 0):
                    Atrain[i][-1]=1
                    sA+=1
            print('Number of classified A points is  ', sA)

            for i in range(lB):
                if(Btrain[i][-1]==0) and (sum(C[j].x*Btrain[i,j] for j in range(l)) -ap.x < 0):
                    Btrain[i][-1]=1
                    sB+=1
            print('Number of classified B points is ', sB)
            #print(Atrain)
            #print(Btrain)


            fuzzy = 0
            for i in range(lA):
                if Atrain[i, -1] == 0:
                    fuzzy += 1

            for i in range(lB):
                if Btrain[i, -1] == 0:
                    fuzzy += 1

            if ap.x >= bp.x or fuzzy == 0:
                itera = 0
                print('-----')
                print('Successful termination with 0 misclassification!')
            elif itera >3 and oldfuzzy[-2] == oldfuzzy[-1] == fuzzy :
                itera =0
                print('------')
                print('Fail to classify : termination with ' , fuzzy, ' misclassifications')
                print('------')
            else :
                oldfuzzy.append(fuzzy)
                itera +=1

########################################TEST#################################################################
Atest = []
Btest = []

Cl = np.array(Cl)
fail = 0
ltst = len(test)

for i in range(ltst) :
    if test[i,0] == 1 :
        Atest.append(test[i])
    else :
        Btest.append(test[i])

Atest = np.array(Atest)
Btest = np.array(Btest)

lA = len(Atest)
lB = len(Btest)
l = len(data[0])-1
c = len(Cl)

ptest = np.zeros((c,lA))
qtest = np.zeros((c,lB))

for i in range(c):
    for j in range(lA):
        for k in range(l):
            ptest[i,j] += Cl[i,k]*Atest[j,k]

for i in range(c):
    for j in range(lB):
        for k in range(l):
            qtest[i,j] += Cl[i,k]*Btest[j,k]
#
w=0

while w<c:
    lA=len(Atest)
    lB=len(Btest)

    rmpostest=[]
    rmnegtest=[]

    for i in range(lA):
        if(ptest[w,i]-apl[w]<0):
            fail += 1
    for i in range(lB):
        if(qtest[w,i]-bpl[w]>0):
            fail+=1
    s=0

    for i in range(lA):
        if(ptest[w,i]-bpl[w]<0):
            rmpostest.append(i)

            if s==0:
                s=1

    for i in range(lB):
        if(qtest[w,i]-apl[w]>0):
            rmnegtest.append(i)

            if s==0:
                s=1

    Atest = Atest[rmpostest]
    Btest = Btest[rmnegtest]

    w+=1

success_rate = (ltst-fail)*100/ltst
print(success_rate)
