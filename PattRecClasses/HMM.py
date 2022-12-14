import numpy as np
from .DiscreteD import DiscreteD
from .GaussD import GaussD
from .MarkovChain import MarkovChain


class HMM:
    """
    HMM - class for Hidden Markov Models, representing
    statistical properties of random sequences.
    Each sample in the sequence is a scalar or vector, with fixed DataSize.
    
    Several HMM objects may be collected in a single multidimensional array.
    
    A HMM represents a random sequence(X1,X2,....Xt,...),
    where each element Xt can be a scalar or column vector.
    The statistical dependence along the (time) sequence is described
    entirely by a discrete Markov chain.
    
    A HMM consists of two sub-objects:
    1: a State Sequence Generator of type MarkovChain
    2: an array of output probability distributions, one for each state
    
    All states must have the same class of output distribution,
    such as GaussD, GaussMixD, or DiscreteD, etc.,
    and the set of distributions is represented by an object array of that class,
    although this is NOT required by general HMM theory.
    
    All output distributions must have identical DataSize property values.
    
    Any HMM output sequence X(t) is determined by a hidden state sequence S(t)
    generated by an internal Markov chain.
    
    The array of output probability distributions, with one element for each state,
    determines the conditional probability (density) P[X(t) | S(t)].
    Given S(t), each X(t) is independent of all other X(:).
    
    
    References:
    Leijon, A. (20xx) Pattern Recognition. KTH, Stockholm.
    Rabiner, L. R. (1989) A tutorial on hidden Markov models
    	and selected applications in speech recognition.
    	Proc IEEE 77, 257-286.
    
    """
    def __init__(self, mc, distributions):

        self.stateGen = mc
        self.outputDistr = distributions

        self.nStates = mc.nStates
        self.dataSize = distributions[0].dataSize
    
    def rand(self, nSamples):
        """
        [X,S]=rand(self,nSamples); generates a random sequence of data
        from a given Hidden Markov Model.
        
        Input:
        nSamples=  maximum no of output samples (scalars or column vectors)
        
        Result:
        X= matrix or row vector with output data samples
        S= row vector with corresponding integer state values
          obtained from the self.StateGen component.
          nS= length(S) == size(X,2)= number of output samples.
          If the StateGen can generate infinite-duration sequences,
              nS == nSamples
          If the StateGen is a finite-duration MarkovChain,
              nS <= nSamples
        """
        
        #*** Insert your own code here and remove the following error message
        
        S = self.stateGen.rand(nSamples)
        nS = len(S)
        
        X = np.zeros([self.dataSize, nS])
        for t in range(nS):
            X[:,t:t+1] = self.outputDistr[S[t]].rand(1)
            
        return X, S
        
    def viterbi(self):
        pass

    def train(self, data, data_num):
        
#         q0 A0 B0 initialization should be done when created this HMM class
        max_round = 4
        
        R = data_num
        
        q_iterate = self.stateGen.q     #InitialProb(i)= P[S(1) = i]
        A_iterate = self.stateGen.A     #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]
#         print(q_iterate)
#         print(A_iterate)
        II = A_iterate.shape[0]
        JJ = A_iterate.shape[1]
#         print(II,JJ)
        temp_logprob = 0
        
        for i in range(max_round):
            ############# Baum-Welch ####################
            gamma_bar = np.zeros(self.nStates)
            zeta_bar = np.zeros([II,JJ])

            
            ####### Same as the GetPx function ##########
            for r in range(R):
            
                N = len(self.outputDistr)
                T = data[r].shape[1]
                
                scaled_factor = np.zeros(T)
                pX = np.zeros([N,T])
                
                data_r = np.array(data[r])
#                 print(data_r)
                for n in range(N):
                    pX[n,:] = self.outputDistr[n].prob(data_r)

                for t in range(T):
                    scaled_factor[t] = max(pX[:,t])
                    pX[:,t] = pX[:,t]/max(pX[:,t])
                    
#                 print('pX:',pX)
                ####### Get forward results ##########
                alphahat_matrix, c  = self.stateGen.forward(pX)
#                 print('c',c)
                ####### Get backward results ##########
                betahat_matrix = self.stateGen.backward(pX, c)
                
                ########### Gamma for updating q and b #######
                gamma_matrix = np.zeros([N,T])
                gamma_matrix = alphahat_matrix * betahat_matrix * c[0:T]
                gamma_bar = gamma_bar+gamma_matrix[:,0]
#                 print('gamma_matrix',gamma_matrix)
#                 print('alphahat_matrix',alphahat_matrix)
#                 print('betahat_matrix',betahat_matrix)
                ########### zeta for updating aij ########
                ###per r 
                zeta_matrix = np.zeros([II,JJ])
                
                
                for tt in range(T-1):
                    
                    for ii in range(II):
                        for jj in range(JJ):
                            try:
                                zeta_matrix[ii,jj] = alphahat_matrix[ii,tt] * A_iterate[ii,jj] * pX[jj,tt+1] * betahat_matrix[jj,tt+1]
                            except: 
                                zeta_matrix[ii,jj] = 0
                                
                    zeta_bar = zeta_bar + zeta_matrix
                
#                 print('zeta_bar',zeta_bar)
                ########### Gaussian Update ########
                
                mean_bar = np.zeros((len(self.outputDistr), *np.array(self.outputDistr[0].means.shape)))
#                 print(mean_bar)
                for ii in range(II):
                    
                    for tt in range(T):
#                         print(ii,tt,mean_bar.shape,gamma_matrix.shape,data_r.shape)
#                         print(mean_bar[ii])
                        mean_bar[ii] = mean_bar[ii] + gamma_matrix[ii, tt] * data_r[:, tt]

                    if gamma_matrix[ii,:].sum()!= 0:
                        self.outputDistr[ii].means = mean_bar[ii]/gamma_matrix[ii,:].sum()
                        
#                     print('mean:',self.outputDistr[ii].means)
                
                cov_bar = np.zeros((len(self.outputDistr),*self.outputDistr[0].cov.shape))
#                 print(cov_bar)
                for ii in range(II):
                    
                    for tt in range(T):
                        cov_bar[ii] = cov_bar[ii] + gamma_matrix[ii, tt] * ((data_r[:, tt] - self.outputDistr[ii].means)@(data_r[:, tt].T - self.outputDistr[ii].means.T)) ####???????????? ?????????????????????1?????????????????????
                
                    if gamma_matrix[ii,:].sum()!= 0:    
                        self.outputDistr[ii].cov = cov_bar[ii]/gamma_matrix[ii,:].sum() + np.eye(self.outputDistr[0].cov.shape[0])*0.00001
#                     print('cov:',self.outputDistr[ii].cov)

            if sum(gamma_bar)!=0:
                q_iterate = gamma_bar/sum(gamma_bar)
            
            self.stateGen.q = q_iterate
#             print('q_iterate:',q_iterate)
            
            for ii in range(II):
                for jj in range(JJ):
                    if np.sum(zeta_bar[ii,:])!=0:
                        A_iterate[ii,jj] = zeta_bar[ii,jj]/np.sum(zeta_bar[ii,:])  ####???????????????
#                     print('A_iterate[ii,jj]',A_iterate[ii,jj])
            
            self.stateGen.A =  A_iterate
            
#             print('A_iterate:',A_iterate)
            check = self.logprob(data_r)
            print('logprob:',check,' ite:', i+1)
            if i>0:
                if check-temp_logprob<1:
                    print('prob_change:',check-temp_logprob,'terminate after ite:', i+1)
                    break
                else:
                    temp_logprob = check
            else:
                temp_logprob = check

            
        return None

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass

    def logprob(self, data):
        
        ####### Same as the GetPx function ##########
        N = len(self.outputDistr)
        T = data.shape[1]
        scaled_factor = np.zeros(T)
        pX = np.zeros([N,T])
        for n in range(N):
            pX[n,:] = self.outputDistr[n].prob(data)

        for t in range(T):
            scaled_factor[t] = max(pX[:,t])
            pX[:,t] = pX[:,t]/max(pX[:,t])
        
        ####### Get forward results ##########
        alphahat_matrix, c  = self.stateGen.forward(pX)
        
        ###### Scaling back the c vector except the terminating one (T+1) ##########
        c[0:T] = c[0:T] * scaled_factor
        
        ######## Likelihood ##########
        ln_prob_output_giventhismc = sum(np.log(c))
        
        return ln_prob_output_giventhismc

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass