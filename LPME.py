import numpy as np
from numpy.polynomial import polynomial as poly
import pandas as pd


#functions to generate random keys
def polymul(x, y, modulus, poly_mod):
    return np.int64(np.round(poly.polydiv(poly.polymul(x, y) % modulus, poly_mod)[1] % modulus))

def polyadd(x, y, modulus, poly_mod):
    return np.int64(np.round(poly.polydiv(poly.polyadd(x, y) % modulus, poly_mod)[1] % modulus))

def gen_binary_poly(size):
    #return np.random.randint(0, 2, size, dtype=np.int64)
    sk = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    sk = np.asarray(sk)
    return sk
    

def gen_uniform_poly(size, modulus):
    #return np.random.randint(0, modulus, size, dtype=np.int64)
    a = [33069, 14354, 61150, 5079, 4813, 61939, 48212, 15209, 1916, 64176, 15340, 34874,
         23392, 5500, 44638, 8577, 60549, 13446, 2196, 12972, 30699, 60878, 61096, 63875, 25446, 56251, 56956, 61123, 10296, 13870, 3770, 5442]
    a = np.asarray(a)
    return a
    

def gen_normal_poly(size):
    #return np.int64(np.random.normal(0, 2, size=size))
    e = [0, 1, 0, 1, 0, -1, 0, 1, 0, -2, -1, 1, -1, 4, 3, -1, -2, 1, -3, -1, -2, 0, 1, 2, -3, -3, 0, 0, 0, -1, 0, 2]
    e = np.asarray(e)
    return e
    
#function to generate random key
def keygen(size, modulus, poly_mod):
    sk = gen_binary_poly(size)
    temp = sk.tolist()
    print(temp)
    a = gen_uniform_poly(size,modulus)
    temp1 = a.tolist()
    print(temp1)
    e = gen_normal_poly(size)
    temp2 = e.tolist()
    print(temp2)
    b = polyadd(polymul(-a, sk, modulus, poly_mod), -e, modulus, poly_mod)
    return (b, a), sk

#function to encrypt data
def encrypt(pk, size, q, t, poly_mod, pt):
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m  % q
    e1 = gen_normal_poly(size)
    e2 = gen_normal_poly(size)
    u = gen_binary_poly(size)
    ct0 = polyadd(polyadd(polymul(pk[0], u, q, poly_mod), e1, q, poly_mod), scaled_m, q, poly_mod)
    ct1 = polyadd(polymul(pk[1], u, q, poly_mod), e2, q, poly_mod)
    return (ct0, ct1)

#function to decrypt data with privacy
def decrypt(sk, size, q, t, poly_mod, ct):
    scaled_pt = polyadd(polymul(ct[1], sk, q, poly_mod), ct[0], q, poly_mod)
    decrypted_poly = np.round(scaled_pt * t / q) % t
    return int(decrypted_poly[0])

def add_plain(ct, pt, q, t, poly_mod):
    size = len(poly_mod) - 1
    # encode the integer into a plaintext polynomial
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m  % q
    new_ct0 = polyadd(ct[0], scaled_m, q, poly_mod)
    return (new_ct0, ct[1])

def mul_plain(ct, pt, q, t, poly_mod):
    size = len(poly_mod) - 1
    # encode the integer into a plaintext polynomial
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    new_c0 = polymul(ct[0], m, q, poly_mod)
    new_c1 = polymul(ct[1], m, q, poly_mod)
    return (new_c0, new_c1)

def privacyPreservingTrain(inputFile,outputFile):
    n = 2**5
    # ciphertext modulus
    q = 2**16
    # plaintext modulus
    t = 2**10
    # polynomial modulus
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    # Keygen
    pk, sk = keygen(n, q, poly_mod)
    data = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n'
    dataset = pd.read_csv(inputFile)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    for i in range(len(dataset)):
        temp = []
        for j in range(0,13):
            value = dataset[i,j]
            enc = encrypt(pk, n, q, t, poly_mod, value)
            encryptData = enc[0]
            encryptData = encryptData[0]
            data+=str(encryptData)+","
        data+=str(int(dataset[i,13]))    
        data+="\n"
    f = open(outputFile, "w")
    f.write(data)
    f.close()   

def privacyPreservingTest(inputFile,outputFile):
    n = 2**5
    # ciphertext modulus
    q = 2**16
    # plaintext modulus
    t = 2**10
    # polynomial modulus
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    # Keygen
    pk, sk = keygen(n, q, poly_mod)
    data = 'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n'
    dataset = pd.read_csv(inputFile)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    for i in range(len(dataset)):
        temp = []
        for j in range(0,13):
            value = dataset[i,j]
            enc = encrypt(pk, n, q, t, poly_mod, value)
            encryptData = enc[0]
            encryptData = encryptData[0]
            data+=str(encryptData)+","
        data = data[0:len(data)-1]    
        data+="\n"
    f = open(outputFile, "w")
    f.write(data)
    f.close()        

'''
ct2 = encrypt(pk, n, q, t, poly_mod, pt2)

data = ct1[0]
data = data[0]
print(data)

print("[+] Ciphertext ct1({}):".format(pt1))
print("")
print("\t ct1_0:", ct1[0])
print("\t ct1_1:", ct1[1])
print("")
print("[+] Ciphertext ct2({}):".format(pt2))
print("")
print("\t ct1_0:", ct2[0])
print("\t ct1_1:", ct2[1])
print("")

# Evaluation
ct3 = add_plain(ct1, cst1, q, t, poly_mod)
ct4 = mul_plain(ct2, cst2, q, t, poly_mod)

# Decryption
decrypted_ct3 = decrypt(sk, n, q, t, poly_mod, ct3)
decrypted_ct4 = decrypt(sk, n, q, t, poly_mod, ct4)

print("[+] Decrypted ct3(ct1 + {}): {}".format(cst1, decrypted_ct3))
print("[+] Decrypted ct4(ct2 * {}): {}".format(cst2, decrypted_ct4))
'''

