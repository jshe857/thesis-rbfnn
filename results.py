#!/usr/bin/python2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rc('text',usetex=True)
plt.rc('font', family='serif') #Boston Housing Overfit
# sns.set_style("whitegrid") # plt.style.use('fivethirtyeight')
# plt.style.use('bmh')
# plt.style.use('ggplot')
sns.set_context('paper')



ep_train = np.array([4.1272025848,4.1273207546,4.184813522,4.1932829004,4.1575721302,4.1841846921])
ep_test = np.array([4.5186761294,4.5060368288,4.5722698068,4.5969571842,4.5353319837,4.5723903096])
em_test = np.array([8.0810214418,8.4227555651,8.1442880973,7.9887169239,7.9161632818,8.0810214418])
em_test = np.array([8.0810214418,7.4227555651,8.1442880973,8.9887169239,8.9161632818,9.0810214418])
em_train = np.array([9.2513305486,8.281245326,7.6613469048,6.7674878493,6.5009444502,6.459586032])
#ep_avg = 0.5*(ep_train+ep_test)
#em_avg = 0.5*(em_train+em_test)

ep_overfit = np.array([0.0948520303,0.0917583335,0.0925862724,0.0962668853,0.0908606854,0.0927792739])*100
em_overfit = np.array([-0.126501707,0.0170880385,0.063036069,0.1804553036,0.2176943431,0.2510122788])*100
n = [30,50,60,75,100,125]
plt.figure(dpi=100)
plt.title('Boston Housing Dataset')
# plt.tight_layout()
# plt.subplot(2,1,1)
plt.plot(n,em_test,'o-',label="EM-test")
plt.plot(n,em_train,'o-',label="EM-train")
plt.plot(n,ep_test,'o-',label="EP-test")
plt.plot(n,ep_train,'o-',label="EP-train")
plt.plot(50,em_test[1],'ro',label="Optimal Setting")
plt.plot(50,ep_test[1],'ro',)
plt.plot(50,em_train[1],'ro')
plt.plot(50,ep_train[1],'ro')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
          ncol=5, fancybox=True, shadow=True)
# plt.xlabel("Network Size")
plt.ylabel("RMSE")
plt.xlabel("Network Size")
plt.title("Prediction Error vs. Network Size")
# plt.plot(50,0.0917583335*100,'o')

# plt.subplot(2,1,2)
# plt.plot(n,ep_overfit,'o-',label="EP")
# plt.plot(n,em_overfit,'o-',label="EM")
# # plt.plot(50,0.0170880385*100,'o')
# #plt.plot(n,np.zeros(len(n)),'--',label="Optimal")
# optimal = plt.axhline(y=0,color="black",linewidth="2",label="Optimal")
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
          # ncol=3, fancybox=True, shadow=True)
# plt.xlabel("Network Size")
# plt.ylabel(r"$\Delta$ Test Error/ Training Error \%")
# plt.title("Level of Overfitting vs. Network Size")
# plt.savefig("boston_overfit.png")

#########################POSTERIOR ESTIMATION###############################


# updated m __str__ = [[ 0.09054334]]
# updated m __str__ = [[ 2.33541186 -0.00947378]]
# updated v __str__ = [[ 0.141674    0.01190881]]
# updated v __str__ = [[ 0.00302139]]
def normpdf(m,v,x):
   
   return np.exp(-(x-m)**2/(2*v)) / (np.sqrt(2*np.pi*v))

def plot_posterior(csv,ep_m,ep_v,title):
    plt.figure()
    n, bins, patches = plt.hist(csv,80, normed=True,histtype='stepfilled',label="MCMC")
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    print ep_m
    for m,v in zip(ep_m,ep_v):
        x = np.linspace(m-5*np.sqrt(v),m+5*np.sqrt(v),num=200)
        plt.plot(x,normpdf(m,v,x),'b-',label="EP",alpha=0.75)
    plt.legend()
    plt.title(title)


csv = np.genfromtxt ('C_1d_1network.txt', delimiter=",",skip_header=0)
plot_posterior(csv,[0.09054334],[0.00000302139],'C')


csv = np.genfromtxt ('w_1d_1network.txt', delimiter=",",skip_header=0)
plot_posterior(csv,[-0.00947378,2.13541186],[0.00001190881,0.000141674],'w')

csv = np.genfromtxt ('C_1d_1network_20.txt', delimiter=",",skip_header=0)
plot_posterior(csv,[0.09054334],[0.00000302139],'C')


csv = np.genfromtxt ('w_1d_1network_20.txt', delimiter=",",skip_header=0)
plot_posterior(csv,[-0.00947378,2.13541186],[0.00001190881,0.000141674],'w')

#########################Uncertainty Estimate###############################
def generate_xy(rng,num,noise=True):
    x_pts =  np.linspace(-rng,rng,num=num)
    X = np.array([x_pts]).T
    if (noise):
        y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15) + 0.3*np.random.randn(num)
        #y = 2*np.exp(-10*(x_pts - 0.1)**2) + 0.1*np.random.randn(num) 
    else:
        y = 3*np.cos(x_pts/9) +  2*np.sin(x_pts/15)
        #y = 2*np.exp(-10*(x_pts - 0.1)**2)
    return(X,y)

fig,ax = plt.subplots(2, sharex=True,figsize=(7,8))
plt.suptitle('Variance')
rng = 50
x_true,y_true = generate_xy(rng,200,noise=False)
x_noise,y_noise = generate_xy(rng,100,noise=True)


 

csv = np.genfromtxt ('1d_variance_100.txt', delimiter=",",skip_header=0)
x_pts =  np.linspace(-50,50,num=500)
ax[0].fill_between(x_pts,csv[:,0],csv[:,2],alpha=0.2) 
ax[0].plot(x_pts,csv[:,1],'b-',label='Prediction') 
ax[0].plot(x_true,y_true,'g-',label='True function')
ax[0].plot(x_noise,y_noise,'r.',label='Training data')
ax[0].legend()
ax[0].set_title('n = 100')
ax[0].set_ylabel('y')

csv = np.genfromtxt ('1d_variance_500.txt', delimiter=",",skip_header=0)
x_pts =  np.linspace(-50,50,num=500)
ax[1].fill_between(x_pts,csv[:,0],csv[:,2],alpha=0.2) 
ax[1].plot(x_pts,csv[:,1],'b-',label='Prediction') 
ax[1].plot(x_true,y_true,'g-',label='True function')
ax[1].plot(x_noise,y_noise,'r.',label='Training data')
ax[1].legend()
ax[1].set_ylabel('y')
ax[1].set_xlabel('x')
ax[1].set_title('n = 500')
plt.show()
