

# DWM
## Exp 6
```python
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas
import matplotlib
from pandas.plotting import scatter_matrix
```
```python
data = sns.load_dataset("iris")
data.head(8)
```
```python
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
```
```python
plt.xlabel('Features')
plt.ylabel('Species')
```
```python
pltX = data.loc[:,'sepal_length']
pltY = data.loc[:,'species']
plt.scatter(pltX,pltY,color='blue',label='sepal_length')
```
```python
pltX = data.loc[:,'sepal_width']
pltY = data.loc[:,'species']
plt.scatter(pltX,pltY,color='green',label='sepal_width')
```
```python
pltX = data.loc[:,'petal_length']
pltY = data.loc[:,'species']
plt.scatter(pltX,pltY,color='red',label='petal_length')
```
```python
pltX = data.loc[:,'petal_width']
pltY = data.loc[:,'species']
plt.scatter(pltX,pltY,color='black',label='petal_width')

plt.legend(loc=4, prop={'size':8})
plt.show()
```
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
```python
#Train the model
model = LogisticRegression()
model.fit(x_train, y_train) #Training the model

#Test the model
predictions = model.predict(x_test)
print(predictions)# printing predictions

print()# Printing new line
```
```python
#Check precision, recall, f1-score
print( classification_report(y_test, predictions) )

print( accuracy_score(y_test, predictions))

scatter_matrix(data)
plt.show()
```

## Exp 7
```python
#Importing libraries
from ast import increment_lineno
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
```
```python
iris = load_iris()
data = sns.load_dataset("iris")
data.head(8)
```
```python
x = data.iloc[:, :-1]
df = pd.DataFrame(data = iris.data, columns = ['sepal length','sepal width','petal length','petal width'])
df['target']=pd.Series(iris.target)
df['target_names']=pd.Series(iris.target_names)
species = []
for i in range(len(df)):
    if df.iloc[i]['target'] == 0:
        species.append('setosa')
    elif df.iloc[i]['target'] == 1:
        species.append('versicolor')
    elif df.iloc[i]['target'] == 2:
        species.append('virginica')
df['Species'] = species
df
```
```python
plt.scatter(x=df['sepal length'], y=df['sepal width'], c=iris.target, cmap='gist_rainbow')
plt.xlabel('Sepal Width', fontsize=18)
plt.ylabel('Sepal Length', fontsize=18)
```
```python
sns.pairplot(df.drop(['target'], axis=1), hue='Species', height=2.5, markers = ["8","s","D"])
kmeans5 = KMeans(n_clusters=5, init = 'k-means++', random_state = 0)
y = kmeans5.fit_predict(x)
print(y)
```
```python
y_means = kmeans5.fit_predict(x)
X = np.array(x)
plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50, c='red', label='Cluster_1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50, c='blue', label='Cluster_2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50, c='green', label='Cluster_3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50, c='cyan', label = 'Cluster_4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=50, c='magenta', label = 'Cluster_5')
```
```python
plt.scatter(kmeans5.cluster_centers_[:,0], kmeans5.cluster_centers_[:,1], s=25, c='yellow', label='C')
plt.legend()
plt.show()
```
```python
Error =[]
for i in range(1, 11):
    kmeans11 = KMeans(n_clusters = i, init = 'k-means++', n_init=10, max_iter = 300, random_state = 0)
    kmeans11.fit(x)
    Error.append(kmeans11.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,11),Error)
plt.title('Elbow_Method using WCSS with k=1-11')
plt.xlabel('No. of clusters')
plt.ylabel('Error')
plt.show()
```
```python
kmeans3 = KMeans(n_clusters = 3, random_state=21)
y = kmeans3.fit_predict(x)
print(y)
kmeans3.cluster_centers_
```
```python
from pylab import *
rcParams['figure.figsize'] = 15,5
fig, axes = plt.subplots(1, 2,)
axes[0].scatter(x=df['sepal length'], y=df['sepal width'], c=y, cmap='gist_rainbow', edgecolor='k',label='length x width')
axes[1].scatter(x=df['sepal length'], y=df['sepal width'], c=iris.target, cmap='jet', edgecolor='k', label='length x width')
axes[0].scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1], s=180, c='yellow', label='x')
axes[1].scatter(kmeans3.cluster_centers_[:, 0], kmeans3.cluster_centers_[:, 1], s=180, c='yellow', label='y')
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
```
```python
from pylab import *
rcParams['figure.figsize'] = 15,5
fig, axes = plt.subplots(1, 2,)
axes[0].scatter(x=df['petal length'], y=df['petal width'], c=y, cmap='gist_rainbow', edgecolor='k',label='x')
axes[1].scatter(x=df['petal length'], y=df['petal width'], c=iris.target, cmap='jet', edgecolor='k',label='y')
axes[0].set_xlabel('petal length', fontsize=18)
axes[0].set_ylabel('petal width', fontsize=18)
axes[1].set_xlabel('petal length', fontsize=18)
axes[1].set_ylabel('petal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
```
```python
pd.crosstab(iris.target,y)
```

## Exp 8
```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset =[['milk','onion','nutmeg','kidney beans','eggs','yogurt'],
['dill','onion','nutmeg','kidney beans','eggs','yogurt'],
['milk','apple','kidney beans','eggs'],
['milk','unicorn','corn','kidney beans','yogurt'],
['corn','onion','onion','kidney beans','ice cream','eggs'],
]
```
```python
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df
```
```python
frequent_itemsets = apriori(df,min_support=0.6,use_colnames=True)
frequent_itemsets
```
```python
res = association_rules(frequent_itemsets, metric='confidence',min_threshold=0.7)
res
```
```python
res1 = res[['antecedents','consequents','support','confidence','lift']]
res1
```
```python
res2= res1[res['confidence'] >=1]
res2
```

## Exp 9
```python
import numpy as np
import numpy.linalg as la
np.set_printoptions(suppress=True)
```
```python
L = np.array([[0, 1/2, 1/3, 0, 0, 0],
[1/3, 0, 0, 0, 1/2, 0],
[1/3, 1/2, 0, 1, 0, 1/2],
[1/3, 0, 1/3, 0, 1/2, 1/2],
[0, 0, 0, 0, 0, 0],
[0, 0, 1/3, 0, 0, 0]])
```
```python
eVals, eVecs = la.eig(L)
order = np.absolute(eVals).argsort()[::-1]
eVals = eVals[order]
eVecs = eVecs[:,order]
r = eVecs[:,0]
100 * np.real(r / np.sum(r))
```
```python
r = 100 * np.ones(6) / 6
r
```
```python
for i in np.arange(100):
    r = L @ r
r
```
```python
r = 100 * np.ones(6) / 6
lastR = r
r = L @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L @ r
    i += 1
print(str(i) + " Iterations to convergence.")
r
```
```python
L2 = np.array([[0, 1/2, 1/3, 0, 0, 0, 0],
[1/3, 0, 0, 0, 1/2, 0, 0],
[1/3, 1/2, 0, 1, 0, 0, 0],
[1/3, 0, 1/3, 0, 1/2, 0, 0],
[0, 0, 0, 0, 0, 0, 0],
[0, 0, 1/3, 0, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 1]])
r = 100 * np.ones(7) / 7
lastR = r
r = L2 @ r
i = 0
while la.norm(lastR - r) > 0.01 :
    lastR = r
    r = L2 @ r
    i += 1
print(str(i) + " Iterations to convergence.")
r
```
```python
def pageRank(linkMatrix, d):
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1-d)/n * np.ones([n,n])
    r = 100 * np.ones(n) / n
    last = r
    r = M @ r
    while la.norm(last - r) > 0.01 :
        last = r
        r = M @ r
    return r
def generate_internet(n):
    c = np.full([n,n], np.arange(n))
    c = (abs(np.random.standard_cauchy([n,n])/2)) > (np.abs(c - c.T)) + 0
    c = (c+1e-10) / np.sum((c+1e-10), axis=0)
    return c
generate_internet(5)
```
```python
L = generate_internet(100)
pageRank(L, 1)
```

## Exp 10

```python
import networkx as nx 
import matplotlib.pyplot as plt
```
```python
G = nx.DiGraph()

G.add_edges_from([('A','D'),('B','C'),('B','E'),('C','A'),
                  ('D','C'),('E','D'),('E','B'),('E','F'),
                  ('E','C'),('F','C'),('F','H'),('G','A'),
                  ('G','C'),('H','A') ])

```
```python
plt.figure(figsize=(10,10))
nx.draw_networkx(G,with_labels=True)

hubs, authorities = nx.hits(G, max_iter=50,normalized=True)
```
```python
print("Hub Scores: ",hubs)
print("Authority Scores: ", authorities)
```

# AI
## Tic-Tac-Toe
```python
import numpy as np 
import random 
from time import sleep 

def create_board(): 
    return(np.array([[0, 0, 0], 
                     [0, 0, 0], 
                     [0, 0, 0]])) 

def possibilities(board): 
    l = [] 
    for i in range(len(board)): 
        for j in range(len(board)): 
            if board[i][j] == 0: 
                l.append((i, j)) 
    return(l) 

def random_place(board, player): 
    selection = possibilities(board) 
    current_loc = random.choice(selection) 
    board[current_loc] = player 
    return(board) 

def row_win(board, player): 
    for x in range(len(board)): 
        win = True
        for y in range(len(board)): 
            if board[x, y] != player: 
                win = False
                continue
        if win == True: 
            return(win) 
    return(win) 

def col_win(board, player): 
    for x in range(len(board)): 
        win = True
        for y in range(len(board)): 
            if board[y][x] != player: 
                win = False
                continue
        if win == True: 
            return(win) 
    return(win) 

def diag_win(board, player): 
    win = True
    y = 0
    for x in range(len(board)): 
        if board[x, x] != player: 
            win = False
    if win: 
        return win 
    win = True
    if win: 
        for x in range(len(board)): 
            y = len(board) - 1 - x 
            if board[x, y] != player: 
                win = False
    return win 

def evaluate(board): 
    winner = 0
    for player in [1, 2]: 
        if (row_win(board, player) or
            col_win(board,player) or 
            diag_win(board,player)): 
            winner = player 
    if np.all(board != 0) and winner == 0: 
        winner = -1
    return winner 

def play_game(): 
    board, winner, counter = create_board(), 0, 1
    print(board) 
    sleep(2) 
    while winner == 0: 
        for player in [1, 2]: 
            board = random_place(board, player) 
            print("Board after " + str(counter) + " move") 
            print(board) 
            sleep(2) 
            counter += 1
            winner = evaluate(board) 
            if winner != 0: 
                break
    return(winner) 
print("Winner is: " + str(play_game())) 
```

## 8-Puzzle-problem
```python
class Node:
    def __init__(self,data,level,fval):
        """ Initialize the node with the data, level of the node and the calculated fvalue """
        self.data = data
        self.level = level
        self.fval = fval

    def generate_child(self):
        """ Generate child nodes from the given node by moving the blank space
            either in the four directions {up,down,left,right} """
        x,y = self.find(self.data,'_')
        """ val_list contains position values for moving the blank space in either of
            the 4 directions [up,down,left,right] respectively. """
        val_list = [[x,y-1],[x,y+1],[x-1,y],[x+1,y]]
        children = []
        for i in val_list:
            child = self.shuffle(self.data,x,y,i[0],i[1])
            if child is not None:
                child_node = Node(child,self.level+1,0)
                children.append(child_node)
        return children
        
    def shuffle(self,puz,x1,y1,x2,y2):
        """ Move the blank space in the given direction and if the position value are out
            of limits the return None """
        if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):
            temp_puz = []
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None

    def copy(self,root):
        """ Copy function to create a similar matrix of the given node"""
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp    
            
    def find(self,puz,x):
        """ Specifically used to find the position of the blank space """
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data)):
                if puz[i][j] == x:
                    return i,j


class Puzzle:
    def __init__(self,size):
        """ Initialize the puzzle size by the specified size,open and closed lists to empty """
        self.n = size
        self.open = []
        self.closed = []

    def accept(self):
        """ Accepts the puzzle from the user """
        puz = []
        for i in range(0,self.n):
            temp = input().split(" ")
            puz.append(temp)
        return puz

    def f(self,start,goal):
        """ Heuristic Function to calculate hueristic value f(x) = h(x) + g(x) """
        return self.h(start.data,goal)+start.level

    def h(self,start,goal):
        """ Calculates the different between the given puzzles """
        temp = 0
        for i in range(0,self.n):
            for j in range(0,self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp
        

    def process(self):
        """ Accept Start and Goal Puzzle state"""
        print("Enter the start state matrix \n")
        start = self.accept()
        print("Enter the goal state matrix \n")        
        goal = self.accept()

        start = Node(start,0,0)
        start.fval = self.f(start,goal)
        """ Put the start node in the open list"""
        self.open.append(start)
        print("\n\n")
        while True:
            cur = self.open[0]
            print("")
            print("  | ")
            print("  | ")
            print(" \\\'/ \n")
            for i in cur.data:
                for j in i:
                    print(j,end=" ")
                print("")
            """ If the difference between current and goal node is 0 we have reached the goal node"""
            if(self.h(cur.data,goal) == 0):
                break
            for i in cur.generate_child():
                i.fval = self.f(i,goal)
                self.open.append(i)
            self.closed.append(cur)
            del self.open[0]

            """ sort the opne list based on f value """
            self.open.sort(key = lambda x:x.fval,reverse=False)


puz = Puzzle(3)
puz.process()
```

## Travelling Sales Person Problem

```python
from sys import maxsize 

from itertools import permutations

V = 4

def travellingSalesmanProblem(graph, s): 
    vertex = [] 
    for i in range(V): 
        if i != s: 
            vertex.append(i) 
    min_path = maxsize 
    next_permutation=permutations(vertex)
    for i in next_permutation:
        current_pathweight = 0
        k = s 
        for j in i: 
            current_pathweight += graph[k][j] 
            k = j 
        current_pathweight += graph[k][s] 
        min_path = min(min_path, current_pathweight) 
    return min_path 

if __name__ == "__main__": 
    graph = [[0, 10, 15, 20], [10, 0, 35, 25], 

            [15, 35, 0, 30], [20, 25, 30, 0]] 

    s = 0
    print(travellingSalesmanProblem(graph, s))
```

## Water-jug Problem

```python
from collections import defaultdict 

jug1, jug2, aim = 4, 3, 2
visited = defaultdict(lambda: False) 

def waterJugSolver(amt1, amt2):  

    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0): 
        print(amt1, amt2) 
        return True

    if visited[(amt1, amt2)] == False: 
        print(amt1, amt2) 
        visited[(amt1, amt2)] = True
        return (waterJugSolver(0, amt2) or
                waterJugSolver(amt1, 0) or
                waterJugSolver(jug1, amt2) or
                waterJugSolver(amt1, jug2) or
                waterJugSolver(amt1 + min(amt2, (jug1-amt1)), 
                amt2 - min(amt2, (jug1-amt1))) or
                waterJugSolver(amt1 - min(amt1, (jug2-amt2)), 
                amt2 + min(amt1, (jug2-amt2)))) 

    else: 
        return False

print("Steps: ") 

waterJugSolver(0, 0) 
```

## Min-Max Problem

```python
#!/usr/bin/env python3

import math

def minimax (curDepth, nodeIndex,
             maxTurn, scores,
             targetDepth):

    # base case : targetDepth reached
    if (curDepth == targetDepth):
        return scores[nodeIndex]

    if (maxTurn):
        return max(minimax(curDepth + 1, nodeIndex * 2,
                    False, scores, targetDepth),
                   minimax(curDepth + 1, nodeIndex * 2 + 1,
                    False, scores, targetDepth))

    else:
        return min(minimax(curDepth + 1, nodeIndex * 2,
                     True, scores, targetDepth),
                   minimax(curDepth + 1, nodeIndex * 2 + 1,
                     True, scores, targetDepth))

# Driver code
scores = []
i = int(input("Number of nodes: "))
print("Enter Nodes: ")
for i in range(0,i):
    scores.append(int(input()))

treeDepth = math.log(len(scores), 2)
print("The optimal value is : ", end = "")
print(minimax(0, 0, True, scores, treeDepth))
```

## Prolog
### Simple
```prolog
likes(sam, salad).
likes(sam, pie).
likes(sam, apples).
likes(sam, whiskey).
likes(sam, anime).
```

### Count
```prolog
count([],0).
count([H|T],N) :- count(T,N1) , N is N1+1.
```

### Greatest of 3 number
```prolog
max(P,Q,R):-P>Q,P>R,write('Larger number is '),write(P).
max(P,Q,R):-P<Q,Q>R,write('Larger number is '),write(Q).
max(P,Q,R):-R>Q,P<R,write('Larger number is '),write(R).
max(P,Q,R):-P=Q,P<R,write('Larger number is '),write(R).
max(P,Q,R):-P<Q,P=R,write('Larger number is '),write(Q).
max(P,Q,R):-Q=R,P>Q,write('Larger number is '),write(P).
max(P,Q,R):-P=Q,P>R,write('Larger numbers are  '),write(P),write(' and '),write(Q).
max(P,Q,R):-P=R,Q<R,write('Larger numbers are  '),write(P),write(' and '),write(R).
max(P,Q,R):-Q=R,P<R,write('Larger numbers are  '),write(R),write(' and '),write(Q).
max(P,Q,R):-P=Q,P=R,write('All numbers are equal '). 
```

### Factorial 
```prolog
factorial(0, 1).
 factorial(N, F) :- 
       N > 0, 
       Prev is N -1, 
       factorial(Prev, R), 
       F is R * N.

```
