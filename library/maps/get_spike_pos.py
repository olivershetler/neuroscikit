import numpy as np

def spikePos(ts, x, y, t, cPost, shuffleSpks, shuffleCounter=True):

    randtime = 0

    if shuffleSpks:

        # create a random sample to shuffle from -20 to 20 (not including 0)
        randsamples = np.asarray([sample_num for sample_num in range(-20, 21) if sample_num != 0])

        if shuffleCounter:
            randtime = 0
        else:
            randtime = np.random.choice(randsamples, replace=False)

            maxts = max(ts)
            ts += randtime

            if np.sign(randtime) < 0:
                for index in range(len(ts)):
                    if ts[index] < 0:
                        ts[index] = maxts + np.absolute(randtime) + ts[index]

            elif np.sign(randtime) > 0:
                for index in range(len(ts)):
                    if ts[index] > maxts:
                        ts[index] = ts[index] - maxts

            ts = np.sort(ts)
    else:
        ts = np.roll(ts, randtime)

    N = len(ts)
    spkx = np.zeros((N, 1))
    spky = np.zeros_like(spkx)
    newTs = np.zeros_like(spkx)
    count = -1 # need to subtract 1 because the python indices start at 0 and MATLABs at 1


    for index in range(N):

        tdiff = (t - ts[index])**2
        tdiff2 = (cPost-ts[index])**2
        m = np.amin(tdiff)
        ind = np.where(tdiff == m)[0]

        m2 = np.amin(tdiff2)
        #ind2 = np.where(tdiff2 == m2)[0]

        if m == m2:
            count += 1
            spkx[count] = x[ind[0]]
            spky[count] = y[ind[0]]
            newTs[count] = ts[index]

    spkx = spkx[:count + 1]
    spky = spky[:count + 1]
    newTs = newTs[:count + 1]

    return spkx.flatten(), spky.flatten(), newTs.flatten(), randtime
