import numpy as np
import matplotlib.pyplot as plt

#Ex: 3
def delta(t, epsilon=0.001):
    return (1 / (epsilon * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * epsilon**2))
def u(t): 
    return .5*(np.sign(t)+1.0)
def r(t): 
    return t*u(t)
fig1, (ax2) = plt. subplots(1,figsize=(8,4))
t=np.linspace(0,9,5001)
A=2
T=1
RC=0.001
label="$ {}\\times u(t-{})$".format(A,T) 
[line] = ax2.plot(t,(u(t)-(r(t)**2)+(2*(r(t-0.5))**2)-(r(t-1)**2)-u(t-4)+(r(t-4)**2)-(2*(r(t-4.5))**2)+(r(t-5))**2),label=label)
#[line] = ax2.plot(t,(u(t)-r(t)+2*r(t-0.5)),label=label)

ax2.set_xlabel('Time [s]',fontsize=12)
ax2.set_ylabel('Signal [dimensionless]',fontsize=12)
ax2.set_title("$ {}\\times u(t-{})$".format(A,T))
ax2.grid()
ax2.legend(loc='best')
plt.tight_layout(pad=2, w_pad=2, h_pad=2.0)
plt.show()