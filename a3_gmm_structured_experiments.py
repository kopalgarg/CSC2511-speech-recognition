from a3_gmm_structured import *

if __name__ == "__main__":

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