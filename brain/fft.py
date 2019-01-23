import numpy as np

def calcFFT(self):
        # calculates the FFT
        if not self.psi_contains_values:
            raise ValueError("Psi does not contain values or Psi was not initialized!")
        else:
            self.psi_hat_array = np.fft.fft2(self.psi_array) /(np.prod(self.paramObj.getResolution()))
            # self.psi_hat_array = np.fft.fftshift(self.psi_hat_array)
            self.psi_hat_contains_values = True
            return self.psi_hat_array

def calcIFFT(self):
        # calculates the inverse FFT
        if not self.psi_hat_contains_values:
            raise ValueError("Psi_hat does not contain values or Psi_hat was not initialized!")
        else:
            self.psi_array = np.fft.ifft2(self.psi_hat_array) * (np.prod(self.paramObj.getResolution()))
            # self.psi_array = np.fft.ifftshift(self.psi_array)
            self.psi_contains_values = True
            return self.psi_array

def calcSlowFFT(self):
        N_x = psi.shape[0]
        N_y = psi.shape[1]
    
        psi_hat = np.zeros((N_x, N_y)) + 0j
        psi_hat_1 = np.zeros((N_x, N_y)) + 0j
        #opti = np.zeros((N_x, N_y))
    
        for k in range (N_x):
            psi_hat[k,:] = np.fft.fft(psi[k,:])
        # psi_hat[:,:] = FFT_vectorized_img(psi[:,:])
    
        for p in range (N_y):
            psi_hat_1[:,p] = np.fft.fft(psi_hat[:,p])
        # psi_hat_1[:,:] = FFT_vectorized_img(psi_hat[:,:])
    
        return psi_hat_1


def calcSlowIFFT(psi):
        N_x = psi.shape[0]
        N_y = psi.shape[1]
    
        psi_hat = np.zeros((N_x, N_y)) + 0j
        psi_hat_1 = np.zeros((N_x, N_y)) + 0j
        #opti = np.zeros((N_x, N_y))
    
        for k in range (N_x):
            psi_hat[k,:] = np.fft.ifft(psi[k,:])
        # psi_hat[:,:] = FFT_vectorized_img(psi[:,:])
    
        for p in range (N_y):
            psi_hat_1[:,p] = np.fft.ifft(psi_hat[:,p])
        # psi_hat_1[:,:] = FFT_vectorized_img(psi_hat[:,:])
    
        return psi_hat_1

