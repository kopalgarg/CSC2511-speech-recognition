import os
import numpy as np
import re
import string
import sys 
import fnmatch

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                          
    """
    # number of words in ref
    n = len(r) # nrows
    # number of words in hyp
    m = len(h) # ncols
    # matrix of distances
    R = np.zeros((n, m))
    
    nS, nI, nD=0,0,0

    # intialize 
    R[0] = np.arange(m)
    R[:,0] = np.arange(n)

    for i in range(1, n):
        for j in range(1, m):
            nD = R[i-1, j] # deletion
            nI = R[i, j-1] # insertion
            nS = R[i-1, j-1] # substitution

            if r[i] == h[j]: R[i,j] = R[i-1, j-1]
            else: R[i,j] = min(nD, nI, nS)+1

    i, j = R.shape[0]-1, R.shape[1]-1
    #import pdb; pdb.set_trace()
    sub = 0
    ins  = 0
    dele = 0
    while i > 0 and j > 0:
        if i > 0 and j > 0: values = [R[i-1,j-1], R[i,j-1], R[i-1,j]]
        elif j > 0 and i<=0: values = [np.inf, R[i, j-1], np.inf]
        elif j <=0 and i > 0: values = [np.inf, np.inf, R[i-1, j]]
        else: break
        min_val =min(values)
        index = values.index(min_val)
        if index == 0:
            if R[i-1, j-1] == R[i,j] - 1:
                sub += 1
            i -= 1
            j -= 1
        elif index == 1:
            ins += 1
            j -= 1
        else:
            dele +=1 
            i -=1

    WER = [R[-1,-1]/(R.shape[0] - 2)]

    return (WER, sub, ins, dele)

def preprocess(text):
    punc = string.punctuation.replace("[]", "")
    # remove punctuation
    text = text.translate(str.maketrans("", "", punc))
    text = re.sub("<[A-Z]+>", "", text)
    # lower case 
    text = text.lower()
    # strip and split
    text = re.sub(r"[^\w\[\] ]+", "", text).split()[2:]
    return text

if __name__ == "__main__":

    
    werGoogle = []
    werKaldi = []
    #sys.stdout = open('asrDiscussion.txt', 'w')

    for rootdir, dir, files in os.walk(dataDir):
        for speaker in dir:
            print(speaker)
            
            main_file_path = os.path.join(rootdir, speaker)
            
            google_transcripts = os.path.join(main_file_path, "transcripts.Google.txt")
            google_transcripts = open(google_transcripts, 'r').readlines()

            kaldi_transcripts = os.path.join(main_file_path, "transcripts.Kaldi.txt")
            kaldi_transcripts = open(kaldi_transcripts, 'r').readlines()

            real_transcripts = os.path.join(main_file_path, "transcripts.txt")
            real_transcripts = open(real_transcripts, 'r').readlines()
            
            if len(real_transcripts) == 0: print(speaker, " empty reference transcript")
            
            for i, r in enumerate(real_transcripts):
                info_g, info_k = [], []
                r = preprocess(r)
                
                ## Google 
                
                info_g += Levenshtein(r, preprocess(google_transcripts[i]))
                werGoogle.append(info_g[0])

                info_g = [speaker, 'Google', i] + info_g
                s = "{} {} {} {} S:{}, I:{}, D:{}"
                s = s.format(*info_g)
                print(s)

                ## Kaldi
                info_k += Levenshtein(r, preprocess(kaldi_transcripts[i]))
                werKaldi.append(info_k[0])

                info_k = [speaker, 'Kaldi', i] + info_k
                s = "{} {} {} {} S:{}, I:{}, D:{}"
                s = s.format(*info_k)
                print(s)
    werGoogle, werKaldi = np.array(werGoogle), np.array(werKaldi)

    print('Google Mean:', np.mean(werGoogle))
    print('Google STDev:', np.var(werGoogle))



    print('Kaldi Mean:', np.mean(werKaldi))
    print('Kaldi STDev:', np.var(werKaldi))
            

