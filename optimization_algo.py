import matplotlib.pyplot as plt
import numpy as np
x=2*np.random.rand(100,1)
y=4+3*x+np.random.randn(100,1)

plt.plot(x,y,"b.")
plt.show()

#normal equation
x_b=np.c_[np.ones((100,1)),x]
theta_best=np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
#print(theta_best)

x_new=np.array([[0],[2]])
x_new_b=np.c_[np.ones((2,1)),x_new]
y_predict=x_new_b.dot(theta_best)


print(y_predict)

plt.plot(x_new,y_predict,"r-")
plt.plot(x,y,"b.")
plt.axis([0,2,0,15])
plt.show()


from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(x,y)
print(lin_reg.intercept_)
print(lin_reg.coef_)


#batch gradient descent

eta=0.1 # learning rate
n_iteration=1000
m=100
x_b=np.c_[np.ones((100,1)),x]

theta=np.random.randn(2,1) # random initializaion

for iteration in range(n_iteration):
	gradients=2/m*x_b.T.dot(x_b.dot(theta)-y)
	theta=theta-eta*gradients

print(theta)



#stochastic GD

from sklearn.linear_model import SGDRegressor
sgd_reg=SGDRegressor(max_iter=100,tol=1e-3,penalty=None,eta0=0.1)
sgd_reg.fit(x,y.ravel())

print(sgd_reg.intercept_)
print(sgd_reg.coef_)



