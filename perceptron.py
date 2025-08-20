 #the weights produced by this code is similar to the weightsproduced by sklearn perceptron
#we need to consider the perceptrons weight as coefficients to a line , thats why difference in value of my perceptron and sklearn perceptron 
#plotted both came out to be same
def Perceptron(X,y,epochs=1000,learning_rate=0.001):
  X['Bias']=1
  weights=len(X.columns) 
  row_total=len(X)
  initial_weights=np.zeros(weights)
  for i in range(epochs):
    ran=np.random.randint(0,row_total)
    row_select=X.iloc[ran]
    y_pred=y.iloc[ran]
    matrix=np.array(row_select)
    z=np.dot(matrix,initial_weights)
    if z>0:## the activation function sochoosen in this case is a stepfunction
      y_h=1
    else:
      y_h=0
    learned= (y_pred-y_h)*learning_rate*matrix
    initial_weights=initial_weights+learned
  return initial_weights    

