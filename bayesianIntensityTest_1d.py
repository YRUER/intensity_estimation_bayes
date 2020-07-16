import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from theano.tensor.slinalg import cholesky
import sys

def simPP(intensity,bound):
    """
    simulates pp with intensity func intensity and returns sample
    intensity: callable, defined on [0,1]
    bound: upper bound on intensity
    """

    N=np.random.poisson(bound)
    homPP=np.random.uniform(size=N)
    PP=np.array([s for s in homPP if bound*np.random.ranf()<=intensity(s)])

    return PP

def intensityLogGauss(sample,bins,beta):
    """estimates intensity function from sample for bins bins and length param beta,
    returns pymc3-trace
    sample: pp sample, (1,)-numpy array
    bins: number of bins
    beta: beta \in [1/bins,0.5] is recommended
    """

    width=1/bins
    data,edges=np.histogram(sample,bins=bins,range=(0,1))
    distMat=width*np.array([[np.abs(i-j) for i in range(bins)] for j in range(bins)])

    model=pm.Model()

    print('building model')
    with model:

        beta=beta
        sigmaSq=pm.HalfNormal('sigmaSq',5)

        chol=np.sqrt(sigmaSq)*cholesky(pm.math.exp(-1.*distMat**2/(2.*beta**2))+1e-6*np.eye(bins))
        y=pm.Normal('gaussfield',mu=0,sigma=1,shape=bins)

        lam=pm.Deterministic('intensity',width*pm.math.exp(pm.math.dot(chol,y)))
        k=pm.Poisson('points',mu=lam,observed=data)

        print('model built, start sampling')
        trace=pm.sample(draws=1000, tune=500,chains=1)

    #pm.traceplot(trace,varnames=['intensity'])
    #plt.show()

    return trace

def modelOnBetaGrid(sample,bins,N,l,u):
    """intensityLogGauss on a grid of N betavalues, linearly spaced in [l,u],
    returns betaGrid,dataframe with WAIC evaluation (see documentation of pymc3)
    and intensity traces
    """

    betaGrid=np.linspace(l,u,N)
    traces=[]
    WAIC=dict()
    index=0

    for beta in betaGrid:
        trace=intensityLogGauss(sample,bins,beta)
        traces.append(trace['intensity'])
        WAIC[index]=trace
        index+=1

    df=pm.compare(WAIC,ic='WAIC')

    return betaGrid,df,traces

def bestBeta(sample,bins,N,l,u):
    """returns beta with lowest WAIC value from np.linspace(l,u,N),
    see modelOnBetaGrid
    """

    betaGrid,df,traces=modelOnBetaGrid(sample,bins,N,l,u)
    minIndex=df.index[0]

    return betaGrid[minIndex]

#just a simple example:
if __name__ == "__main__":
    bins=5
    width=1/bins
    nrOfModels=4
    grid=np.array([i*width+width/2 for i in range(0,bins)])
    X=np.linspace(0,1,100)

    intensity=np.vectorize(lambda x: 50*np.exp(np.sin(2*np.pi*x)))
    bound=50*3

    PP=simPP(intensity, bound)
    N=PP.size

    Y=intensity(X)
    plt.plot(X,Y,'b-')
    plt.scatter(PP,np.zeros(N),marker='|')
    plt.show()

    cont=input('Continue? y/n ')
    if(cont=='n'):
        sys.exit()

    betaGrid,df,traces=modelOnBetaGrid(PP,bins,nrOfModels,1/bins,0.5)
    print(df)

    fig, ax=plt.subplots(2,int((nrOfModels+1)/2))
    index=0
    for i in [0,1]:
        for j in range(int((nrOfModels+1)/2)):
            ax[i][j].plot(X,Y,'b-',label='intensity')
            ax[i][j].plot(grid,bins*traces[index].mean(axis=0),'r.-',label='beta={:4.2}'.format(betaGrid[index]))
            ax[i][j].plot(grid,np.quantile(bins*traces[index],0.05,axis=0),'r-',linewidth=0.5)
            ax[i][j].plot(grid,np.quantile(bins*traces[index],0.95,axis=0),'r-',linewidth=0.5)
            ax[i][j].scatter(PP,np.zeros(N),marker='|',label='sample')
            ax[i][j].legend(loc='upper right')

            index+=1

    plt.show()
