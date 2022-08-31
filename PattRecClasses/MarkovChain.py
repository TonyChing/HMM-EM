import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]

        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message 
        if self.is_finite == False:
            S = np.zeros(tmax)
            temp_prob = self.q
            for t in range(tmax):
                D = DiscreteD(temp_prob.tolist())
                S[t] = D.rand(1)
                temp_prob = self.A[int(S[t])]
        else:
            S = []
            temp_prob = self.q
            for t in range(tmax):
                D = DiscreteD(temp_prob.tolist())
                S.append(int(D.rand(1)))
                if S[t] >= self.nStates: 
                    S.pop()
#                     print('END state exit at length',int(t+1))
                    break
                temp_prob = self.A[S[t]]
            S = np.array(S)
        return S.astype(int)


    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, output_prob):
        T = output_prob.shape[1]
        
        alphahat_matrix = np.zeros([self.nStates, T])
        if self.is_finite:
            c = np.zeros(T+1)
        else:
            c = np.zeros(T)
        ########### Initialization ###########
        alpha_temp = np.zeros(self.nStates)
        alpha_temp = self.q * output_prob[:,0] ###output_prob N*T each colomn t contains j = 1,2,...,N bj(xt) * is element dot
        
#         for j in range(self.nStates):
#             alpha_temp[j] = self.q[j]*output_prob[j]

        c[0] = sum(alpha_temp)
        alphahat_matrix[:,0] = alpha_temp/c[0]
        
        ########### Loop ###########
        for t in range(1,T):
            temp = alphahat_matrix[:,t-1].T @ self.A[0:self.nStates, 0:self.nStates]
#             for i in range(self.nStates):
#                 temp += alphahat_matrix(:,t-1).T @ self.A
            alpha_temp = output_prob[:,t] * temp
            
            c[t] = sum(alpha_temp)
            alphahat_matrix[:,t] = alpha_temp/c[t]
            
        ########### Termination ###########
        if self.is_finite:
            c[T] = alphahat_matrix[:,T-1].T @ self.A[:,-1]
            
            
     
        
        return  alphahat_matrix, c 

    def finiteDuration(self):
        pass
    
    def backward(self, pX, c):
       
#         _, c = self.forward(pX)
                
        
        beta_hat = np.zeros((np.shape(pX)[0], np.shape(pX)[1]))
        

        # initialization:
        if (self.is_finite):
            beta_hat[:,np.shape(pX)[1]-1] = self.A[:,-1]/(c[len(c)-1]*c[len(c)-2])
        else:
            beta_hat[:,np.shape(pX)[1]-1] = 1/c[len(c)-1]

        # Looping:
        for t in np.arange(1,np.shape(pX)[1]):
            if self.is_finite: 
                beta_hat[:,np.shape(pX)[1]-t-1] = (np.dot(self.A[:,:np.shape(pX)[0]], pX[:,np.shape(pX)[1]-t]*beta_hat[:,np.shape(pX)[1]-t]))/c[len(c)-t-2]
            else:
                beta_hat[:,np.shape(pX)[1]-t-1] = (np.dot(self.A[:,:np.shape(pX)[0]],pX[:,np.shape(pX)[1]-t]*beta_hat[:,np.shape(pX)[1]-t]))/c[len(c)-t-1]
                
        return beta_hat

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
