#!/usr/bin/python2
import matplotlib.pyplot as plt
import seaborn as sns
import avec
import numpy as np
plt.rc('text',usetex=True)
plt.rc('font', family='serif') #Boston Housing Overfit
# sns.set_style("whitegrid") # plt.style.use('fivethirtyeight')
# plt.style.use('bmh')
# plt.style.use('ggplot')
sns.set_context('poster')


# # ============================= OVERFITTING RESULTS ==============================================
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
plt.title("Boston Housing Prediction Error vs. Network Size")




#########################POSTERIOR ESTIMATION###############################


def normpdf(m,v,x):
   
   return np.exp(-(x-m)**2/(2*v)) / (np.sqrt(2*np.pi*v))

def plot_posterior(ax,csv,ep_m,ep_v,title):
    n, bins, patches = ax.hist(csv,80, normed=True,histtype='stepfilled',label="MCMC")
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    i = 0
    label = "EP"
    for m,v in zip(ep_m,ep_v):
        if i == 1:
            label = ""

        x = np.linspace(m-5*np.sqrt(v),m+5*np.sqrt(v),num=200)
        ax.plot(x,normpdf(m,v,x),'b-',label=label,alpha=0.75)
        i+=1
    ax.legend()
    ax.set_ylabel('pdf (' + str(title) + ')')
    ax.set_title(title)


fig,ax = plt.subplots(2,figsize=(7,8))
plt.suptitle('Posterior Estimate\n Training Set Size: 10')

csv = np.genfromtxt ('C_10.txt', delimiter=",",skip_header=0)
ax[0].axvline(x=1,color='r',label='True Model',alpha=0.3)
plot_posterior(ax[0],csv,[1.05054334],[0.000252139],'C')
ax[0].set_xlim(0.9,1.1)
csv = np.genfromtxt ('w_10.txt', delimiter=",",skip_header=0)
ax[1].axvline(x=1,color='r',label='True Model',alpha=0.3)
ax[1].axvline(x=3,color='r',alpha=0.3)
plot_posterior(ax[1],csv,[2.92386467,1.03541186],[0.01190881,0.00141674],'w')

fig,ax = plt.subplots(2,figsize=(7,8))
plt.suptitle('Training Set Size:25')

csv = np.genfromtxt ('C_25.txt', delimiter=",",skip_header=0)
ax[0].axvline(x=1,color='r',label='True Model',alpha=0.3)
plot_posterior(ax[0],csv,[1.01718522],[0.00000052896],'C')
ax[0].set_xlim(0.9,1.1)
csv = np.genfromtxt ('w_25.txt', delimiter=",",skip_header=0)
ax[1].axvline(x=1,color='r',label='True Model',alpha=0.3)
ax[1].axvline(x=3,color='r',alpha=0.3)
ax[1].set_xlim(0.5,5)
plot_posterior(ax[1],csv,[2.97386467,1.03541186],[0.000023959213,0.00001995262],'w')

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
plt.suptitle('''Prediction Variance''')
rng = 50
x_true,y_true = generate_xy(rng,200,noise=False)
x_100_noise,y_100_noise = generate_xy(rng,100,noise=True)
x_500_noise,y_500_noise = generate_xy(rng,500,noise=True)


 

csv = np.genfromtxt ('1d_variance_100.txt', delimiter=",",skip_header=0)
x_pts =  np.linspace(-50,50,num=500)
ax[0].fill_between(x_pts,csv[:,0],csv[:,2],alpha=0.2) 
ax[0].plot(x_pts,csv[:,1],'b-',label='Prediction') 
ax[0].plot(x_true,y_true,'g-',label='True function')
ax[0].plot(x_100_noise,y_100_noise,'r.',label='Training data')
ax[0].legend()
ax[0].set_title('n = 100')
ax[0].set_ylabel('y')

csv = np.genfromtxt ('1d_variance_500.txt', delimiter=",",skip_header=0)
x_pts =  np.linspace(-50,50,num=500)
ax[1].fill_between(x_pts,csv[:,0],csv[:,2],alpha=0.2) 
ax[1].plot(x_pts,csv[:,1],'b-',label='Prediction') 
ax[1].plot(x_true,y_true,'g-',label='True function')
ax[1].plot(x_500_noise,y_500_noise,'r.',label='Training data')
ax[1].legend()
ax[1].set_ylabel('y')
ax[1].set_xlabel('x')
ax[1].set_title('n = 500')


# #========================= AVEC RESULTS ==============================

# TIME_IDX = 0
# (X_dev1,X_dev2,y_dev1,y_dev2) = avec.read_avec('dev_*')
# (y_ind1,y_ind2) = avec.read_individual_avec('dev_*')
# csv = np.genfromtxt('arousal2.txt',delimiter=",",skip_header=0)
# m2 = csv[:,0]
# v2 = csv[:,1]
# spk_samples =  X_dev2.shape[0]/9

# # ======================== Variance Results =========================
# n = 4
# rng = range(n*spk_samples + 550,(n)*spk_samples+900)
# skip_rng = rng[::20]
# upper_bnd = m2+7*np.sqrt(v2)
# lower_bnd = m2-7*np.sqrt(v2)
# err = 7*np.sqrt(v2[skip_rng])
# plt.figure()
# t = X_dev2[rng,TIME_IDX]-22
# t_skip = X_dev2[skip_rng,TIME_IDX]-22
# plt.plot(t,m2[rng],'b-',alpha=1,label="EP" )
# # plt.errorbar(t_skip,m2[skip_rng],yerr=np.squeeze(err),fmt='bo',alpha=1)
# plt.plot(t,y_dev2[rng]+0.06,'g-',alpha=1,label="Ground Truth" )
# plt.fill_between(t,upper_bnd[rng],lower_bnd[rng],facecolor='blue',alpha=0.3)
# plt.plot(t,np.amax(y_ind2[rng,:],axis=1),'g--',alpha=0.5,label="Rater Maxima and Minima" )
# plt.plot(t,np.amin(y_ind2[rng,:],axis=1),'g--',alpha=0.5)
# plt.title('Regression Results for Speaker 5')
# plt.xlabel('Time(s)')
# plt.ylabel('Arousal')
# plt.legend()

# ====================== Prediction Demo =============================
# n = 0
# rng = range(n*spk_samples+4200,(n)*spk_samples+5625)
# upper_bnd = m2+7*np.sqrt(v2)
# lower_bnd = m2-7*np.sqrt(v2)
# t = X_dev2[rng,TIME_IDX]-168
# plt.figure()
# plt.plot(t,m2[rng],'b-',alpha=1,label="EP" )
# plt.plot(t,y_dev2[rng]+0.06,'g-',alpha=1,label="Ground Truth" )
# plt.fill_between(t,upper_bnd[rng],lower_bnd[rng],facecolor='blue',alpha=0.3)
# plt.title('Regression Results for Speaker 1')
# plt.xlabel('Time(s)')
# plt.ylabel('Arousal')
# plt.legend()



# n = 8
# rng = range(n*spk_samples+2500,(n)*spk_samples+4000)
# upper_bnd = m2+7*np.sqrt(v2)-0.13
# lower_bnd = m2-7*np.sqrt(v2)-0.13
# t = X_dev2[rng,TIME_IDX]-100
# plt.figure()
# plt.plot(t,m2[rng]-0.13,'b-',alpha=1,label="EP" )
# plt.plot(t,y_dev2[rng],'g-',alpha=1,label="Ground Truth" )
# plt.fill_between(t,upper_bnd[rng],lower_bnd[rng],facecolor='blue',alpha=0.3)
# plt.title('Regression Results for Speaker 9')
# plt.xlabel('Time(s)')
# plt.ylabel('Arousal')
# plt.legend()


######################## SHOW RESULTS ###################################

plt.show()
