import numpy as np
import matplotlib.pyplot as pl
import pickle as pk

# PART A: Implementt the Discrete Fourier Transform
# comparing this against numpy's np.fft.fft, they don't normalized to 1/M like we do
def discrete_fourier_transform(signal_arr):
    siglen = len(signal_arr)    # M
    Fu = np.zeros(siglen, dtype=np.complex)
    for u in range(Fu.size):
        Fu[u] = np.complex(0, 0)
        for x in range(Fu.size):
            Fu[u] += np.complex(signal_arr[x] * np.cos(-2 * np.pi * u * x / siglen), \
                        signal_arr[x] * np.sin(-2 * np.pi * u * x / siglen))
        Fu[u] /= siglen
    return Fu


# same as function above, just semi-enhanced/optimized
def dft_enhanced(signal_arr):
    siglen = len(signal_arr)
    # tabulate values of cos(2pik/M) and sin(2pik/M)
    cosarr = list(np.cos(2 * np.pi * k / siglen) for k in range(siglen))
    sinarr = list(np.sin(2 * np.pi * k / siglen) for k in range(siglen))

    Fu = np.zeros(siglen, dtype=np.complex)
    for u in range(Fu.size):
        Fu[u] = np.complex(0, 0)
        for x in range(Fu.size):
            k = (u * x) % siglen
            Fu[u] += np.complex(signal_arr[x] * cosarr[k], \
                        signal_arr[x] * -sinarr[k])
        Fu[u] /= siglen
    return Fu


def cmp_cplx(x, y):
    mx = np.square(x.real) + np.square(x.imag)
    my = np.square(y.real) + np.square(y.imag)
    if mx < my:
        return -1
    elif mx == my:
        return 0
    else:
        return 1


def most_sig_freq(signal, n):
    fourier = dft_enhanced(signal)
    fourier = fourier[0: fourier.size/2]    # to not get the repititions
    listfourier = list(fourier)
    listfourier.sort(cmp=cmp_cplx, reverse=True) # gets lower to higher
    most_sig = list()
    for i in range(n):
        item = listfourier[i]
        most_sig.append(item)
    return most_sig


# PART B: Simple sines and cosines
def partb():
    #B.1:
    #sine: f[t] = sin(2*pi*s*t/N) where s=8 and N=128
    #cos: f[t] = cos(2*pi*s*t/N) where s = 8 and N=128
    N = 128
    s = 8
    ftsine = np.zeros(N)
    ftcos = np.zeros(N)
    ftsincossum = np.zeros(N)
    for t in range(N):
        ftsine[t] = np.sin(2 * np.pi * s * t / N)
        ftcos[t] = np.cos(2 * np.pi * s * t / N)
    ftsincossum = np.add(ftsine, ftcos)

    allfts = [(ftsine, 'sine'), 
              (ftcos, 'cosine'),
              (ftsincossum, 'sine + cosine')]
    #Graph problems 1 through 3
    for (ft, name) in allfts:
        #display the waves
        pl.plot(ft)
        pl.title('Original ' + name + ' wave')
        pl.show()

        #apply the dft to this sinusoid and plot the Real, Imaginary,
        #Magnitude, and Phase parts of the result
        fortran = dft_enhanced(ft)

        pl.subplot(2, 2, 1)
        pl.title('Real Part')
        pl.plot(fortran.real)
        pl.subplot(2, 2, 2)
        pl.title('Imag Part')
        pl.plot(fortran.imag)
        pl.subplot(2, 2, 3)
        mag = list(np.sqrt(np.square(n.real) + np.square(n.imag)) for n in fortran)
        pl.title('Magnitude')
        pl.plot(mag)
        pl.subplot(2, 2, 4)
        fortranclean = map(clean, fortran)
        phase = list(np.angle(n) for n in fortranclean) #could also do np.arctan2(n.real, n.imag)
        pl.title('Phase')
        pl.plot(phase)
        pl.show()

    # Graph problem 4 (playing with the relative weightings of the sine and cosine parts)
    weightings = [(0.2, 0.8), (0.4, 0.6), (0.6, 0.4), (0.8, 0.2)]
    for (cosw, sinw) in weightings:
        #display the waves
        ftsincossum = np.add(sinw*ftsine, cosw*ftcos)
        pl.plot(ftsincossum)
        pl.title(str(sinw) + '*sine plus ' + str(cosw) + '*cosine wave')
        pl.show()

        #apply the dft to this sinusoid and plot the Real, Imaginary,
        #Magnitude, and Phase parts of the result
        fortran = dft_enhanced(ftsincossum)

        pl.subplot(2, 2, 1)
        pl.title('Real Part')
        pl.plot(fortran.real)
        pl.subplot(2, 2, 2)
        pl.title('Imag Part')
        pl.plot(fortran.imag)
        pl.subplot(2, 2, 3)
        mag = list(np.sqrt(np.square(n.real) + np.square(n.imag)) for n in fortran)
        pl.title('Magnitude')
        pl.plot(mag)
        pl.subplot(2, 2, 4)
        fortranclean = map(clean, fortran)
        phase = list(np.angle(n) for n in fortranclean)
        pl.title('Phase')
        pl.plot(phase)
        pl.show()
    

def clean(x):
    newreal, newimag = x.real, x.imag
    if np.abs(x.real) < 0.001:
        newreal = 0
    if np.abs(x.imag) < 0.001:
        newimag = 0j
    newcomplex = np.complex(newreal, newimag)
    return newcomplex


def partc(signals):
    rectpuls = signals['rect']
    forrect = dft_enhanced(rectpuls)
    pl.subplot(2, 1, 1)
    pl.title('Magnitude of Rectangular Pulse')
    pl.plot(np.abs(forrect))

    #The Power Spectrum is the square of magnitude (so the sum of squares) of the Fourier coefficients 
    #but normalized by the number of samples.  It's the fraction of the 
    #signal that's at a given frequency.
    pl.subplot(2, 1, 2)
    pl.title('Power spectrum of Rectangular Pulse')
    pl.plot(np.square(np.abs(forrect)) / forrect.size)

    pl.show()


def partd(signals):
    gaussian = signals['gaussian']
    forgaus = dft_enhanced(gaussian)
    pl.subplot(2, 1, 1)
    pl.title('Magnitude of the Gaussian')
    pl.plot(np.abs(forgaus))

    #Power Spectrum
    pl.subplot(2, 1, 2)
    pl.title('Power spectrum of the Gaussian')
    pl.plot(np.square(np.abs(forgaus)) / forgaus.size)

    pl.show()


def parte(signals):
    test = signals['test']
    sig = most_sig_freq(test, 3)
    fortest = dft_enhanced(test)
    print 'Most significant frequencies of \'test\' signal: %s' % str(sig)
    pl.title('Test signal in time domain')
    pl.plot(test)
    pl.show()
    pl.subplot(2, 1, 1)
    pl.title('Test signal in frequency domain real part')
    pl.plot(fortest.real)
    pl.subplot(2, 1, 2)
    pl.title('Test signal in frequency domain imag part')
    pl.plot(fortest.imag)
    pl.show()


def partf(signals):
    #I assumed we could use a built-in inverse dft here
    inputsf = signals['rect']
    outputsg = signals['output']

    Fu = dft_enhanced(inputsf)
    Gu = dft_enhanced(outputsg)
    #Find the transfer function H(u), which transforms F(u) into G(u)
    #That is, G(u) = F(u)*H(u), so H(u) = G(u)/F(u)
    Hu = np.divide(Gu, Fu) #complex division
    #1
    pl.subplot(3, 1, 1)
    pl.title('Magnitude of F(u)')
    pl.plot(np.abs(Fu))
    pl.subplot(3, 1, 2)
    pl.title('Magnitude of transfer function H(u)')
    pl.plot(np.abs(Hu))
    pl.subplot(3, 1, 3)
    pl.title('Magnitude of G(u)')
    pl.plot(np.abs(Gu))
    pl.show()

    #2
    pl.subplot(3, 1, 1)
    pl.title('Rect input signal')
    pl.plot(inputsf)
    ht = np.fft.ifft(Hu)
    pl.subplot(3, 1, 2)
    pl.title('Real part of the inverse Fourier Transfrom of the transfer function')
    pl.plot(ht)
    pl.subplot(3, 1, 3)
    pl.title('Output signal')
    pl.plot(outputsg)
    pl.show()





def main():
    partb()
    signals = pk.load(open('HW4_signals.pkl', 'r'))
    # partc(signals)
    # partd(signals)
    # parte(signals)
    # partf(signals)


if __name__ == "__main__":
    main()