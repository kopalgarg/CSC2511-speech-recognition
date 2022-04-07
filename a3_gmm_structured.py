from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp
import sys

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M 
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))
        self.precomputed = None

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """

        # here I pre-compute terms that are not dependent on x_t

        term_1 = 0.5 * self._d * np.log(2*np.pi)
        term_2 = np.sum(np.square(self.mu)/ (2*self.Sigma), axis = 1)
        term_3 = 0.5 * np.sum(np.log(self.Sigma), axis=1)

        precomputed = (term_1 + term_2 + term_3)[m]
        return precomputed

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    mu, sigma = myTheta.mu[m], myTheta.Sigma[m]
    
    precomputed_terms = myTheta.precomputedForM(m)
    
    term_1 = 0.5 * np.square(x) / sigma
    
    term_2 = mu * x / sigma

    if len(x.shape) == 1:
        x_dependent_term = np.sum(term_1 - term_2)
    else:
        x_dependent_term = np.sum(term_1 - term_2, axis = 1)

    final = -x_dependent_term - precomputed_terms
    return final


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    omega = myTheta.omega
    
    numerator = log_Bs + np.log(omega) 
    denominator = logsumexp(numerator, axis=0)
    
    log_p = numerator - denominator
    return log_p


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    omega = myTheta.omega
    p = log_Bs + np.log(omega)
    to_return = np.sum(logsumexp(p, axis  = 0, keepdims=True))
    return to_return

def computeIntermediaResults(log_Ps, X):
    T = X.shape[0]
    new_omega = np.sum(np.exp(log_Ps), axis = 1)/T
    new_mu = (np.exp(log_Ps) @ X)/np.sum(np.exp(log_Ps), axis = 1)[:, None]
    new_sigma = (np.exp(log_Ps) @ (X**2))/np.sum(np.exp(log_Ps), axis = 1)[:, None]-(new_mu **2)
    
    return new_mu, new_sigma, new_omega

def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)

    # print("TODO : Initialization")

    # for ex.,
    # myTheta.reset_omega(omegas_with_constraints)
    # myTheta.reset_mu(mu_computed_using_data)
    # myTheta.reset_Sigma(some_appropriate_sigma)

    # initialize each mu to a random vector from the data
    random_index= np.random.randint(0, X.shape[0], M)
    random_data = X[random_index]
    myTheta.reset_mu(random_data)

    # initialize sigma to a random diagnoal matrix (or I matrix)
    myTheta.reset_Sigma(np.ones((M, X.shape[1])))

    # intialize omega (sum to 1, and probability so between 0 and 1)
    myTheta.reset_omega(np.ones((M, 1))/M)

    #print("TODO: Rest of training")

    ind = 0
    prev_l, improvement = -np.inf, np.inf

    while ind <= maxIter and improvement >= epsilon:
        log_Bs = np.array([log_b_m_x(m = m,
                                    x = X,
                                    myTheta = myTheta) for m in range(M)])
        log_Ps = log_p_m_x(log_Bs = log_Bs,
                            myTheta = myTheta)
        logLiki = logLik(log_Bs = log_Bs,
                        myTheta = myTheta) 
        new_mu, new_sigma, new_omega =computeIntermediaResults(log_Ps, X)

        improvement = logLiki - prev_l
        prev_l = logLiki
        
        if improvement >= epsilon:
            myTheta.reset_omega(new_omega)
            myTheta.reset_mu(new_mu)
            myTheta.reset_Sigma(new_sigma)
        ind+=1
    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    #print("TODO")
    log_likelihood = []
    #import pdb; pdb.set_trace()
    best_K = -1
    for i, theta in enumerate(models):
        model_name =  theta.name
        log_Bs = np.zeros((models[0].Sigma.shape[0], mfcc.shape[0]))
        for m in range(models[0].Sigma.shape[0]):
            log_Bs[m] = log_b_m_x(m, mfcc, theta)
        likelihood = logLik(log_Bs, theta)
        log_likelihood.append([likelihood, model_name, i])

    if k > 0:
        best_K = k
    else:
        best_K = 0
    
    log_likelihood = sorted(log_likelihood, key=lambda x: x[0], reverse=True)
    log_likelihood = log_likelihood[:best_K]

    bestModel = log_likelihood[0][2]
    
    if k > 0:
        print(models[correctID].name)
        for i in range(k):
            print(log_likelihood[i][1], log_likelihood[i][0])

    return 1 if (bestModel == correctID) else 0

if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    #print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    #=============== For experiments ========================#
    # maxSpeaker_list = [32, 24, 16, 8, 4]
    # M_list = [8, 7, 6, 5, 4, 3, 2, 1]
    # maxIter_list = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
    #=======================================================#

    # train a model for each speaker, and reserve data for testing
    #sys.stdout = open("gmmLikes.txt", "w")
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print("Accuracy: ", accuracy)
    #sys.stdout.close()


    #=============== For experiments ========================#

    '''
    random.seed(0)

    trainThetas = []
    testMFCCs = []

    d = 13
    k = 5  
    epsilon = 0.0
    maxIter = 20
    M = 8

    f = open('gmm_S.txt', 'w')

    maxSpeaker_list = [32, 24, 16, 8, 4]

    print('Experiment 1: MaxSpeaker List')
    for maxSpeaker in maxSpeaker_list:
        s, trainThetas, testMFCCs = 0, [], []
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                print(speaker)
                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)

                if s < maxSpeaker:
                    testMFCCs.append(np.load(os.path.join(dataDir, speaker, files.pop())))

                    X = np.empty((0, d))
                    for file in files:
                        X = np.append(X, np.load(os.path.join(dataDir, speaker, file)), axis=0)
                    trainThetas.append(train(speaker, X, M, epsilon, maxIter))
                else:
                    trainThetas.append(theta(speaker))
                s += 1

        # evaluate
        nc = 0
        for i in range(0, len(testMFCCs)):
            nc += test(testMFCCs[i], i, trainThetas, k)

        accuracy = 1.0 * nc / len(testMFCCs)
        print('Accuracy: ', accuracy)
        print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy))
        print('\n')

        # write to file
        print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy), file=f)



    M_list = [8, 7, 6, 5, 4, 3, 2, 1]
    maxSpeaker=32
    
    print('Experiment 2: M List')
    for M in M_list:
        s, trainThetas, testMFCCs = 0, [], []
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                print(speaker)
                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)

                if s < maxSpeaker:
                    testMFCCs.append(np.load(os.path.join(dataDir, speaker, files.pop())))

                    X = np.empty((0, d))
                    for file in files:
                        X = np.append(X, np.load(os.path.join(dataDir, speaker, file)), axis=0)
                    trainThetas.append(train(speaker, X, M, epsilon, maxIter))
                else:
                    trainThetas.append(theta(speaker))
                s += 1

        # evaluate
        nc = 0
        for i in range(0, len(testMFCCs)):
            nc += test(testMFCCs[i], i, trainThetas, k)

        accuracy = 1.0 * nc / len(testMFCCs)
        print('Accuracy: ', accuracy)
        print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy))
        print('\n')

        # write to file
        print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy), file=f)


    print('Experiment 3: maxIter_list')
    maxIter_list = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
    maxSpeaker=32
    M = 8
    for maxIter in maxIter_list:
        s, trainThetas, testMFCCs = 0, [], []
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                print(speaker)
                files =  fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)

                if s < maxSpeaker:
                    testMFCCs.append(np.load(os.path.join(dataDir, speaker, files.pop())))

                    X = np.empty((0, d))
                    for file in files:
                        X = np.append(X, np.load(os.path.join(dataDir, speaker, file)), axis=0)
                    trainThetas.append(train(speaker, X, M, epsilon, maxIter))
                else:
                    trainThetas.append(theta(speaker))
                s += 1

        # evaluate
        nc = 0
        for i in range(0, len(testMFCCs)):
            nc += test(testMFCCs[i], i, trainThetas, k)

        accuracy = 1.0 * nc / len(testMFCCs)
        print('Accuracy: ', accuracy)
        print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy))
        print('\n')

        # write to file
        print('M: {} \t maxIter: {} \t S: {} \t Accuracy: {}'.format(M, maxIter, maxSpeaker, accuracy), file=f)
    '''