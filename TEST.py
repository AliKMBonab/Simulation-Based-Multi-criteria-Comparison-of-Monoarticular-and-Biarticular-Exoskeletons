import numpy as np
import numpy.polynomial as Polynomial
import matplotlib.pyplot as plt

def f(x):
    y = x**3 - x**2 + 100*np.random.randn(len(x))
    return y

x = np.linspace(-10,10,100)
y = f(x)

coef,prop = np.polynomial.polynomial.polyfit(x,y,deg=1,full=True)
coef_2 = np.polyfit(x,y,1)
print(coef)
print(coef_2)
y_hat = np.polynomial.polynomial.polyval(x,coef)
y_hat_2 = np.polyval(coef_2,x)
plt.figure()
plt.scatter(x,y,color='gray')
plt.plot(x,y_hat,color='r')
plt.plot(x,y_hat_2,color='g')
plt.show()
