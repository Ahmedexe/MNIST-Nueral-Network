import numpy as np
import copy as copy


arr = np.arange(20)
np.random.shuffle(arr)
print(arr)


flarr = []
srarr = []


def mergeSort(arr):
    
    arrL = []
    arrR = []

    if len(arr) != 1:
        n = np.floor(len(arr)/2)
        arrL = arr[0:int(n)]
        arrR = arr[int(n):len(arr) + 1]
        flarr = mergeSort(arrL)
        srarr = mergeSort(arrR)
        if flarr != None:
            arrL = flarr
        if srarr != None:
            arrR = srarr    
    if len(arr) > 1:
        arr = merge(arrL, arrR)
        return arr
    

def merge(arrL, arrR):

    length = len(arrL) + len(arrR)
    final_arr = []
    s = -1000
    for k in range(len(arrR)):
        
        for i in range(len(arrL)): 

            if arrL[i] < arrR[k] and arrL[i] > s:
                s = arrL[i]
                
                final_arr.append(s)
                if len(final_arr) + 1 == len(arrL) + len(arrR):
                    final_arr.append(arrR[k])
            
            elif final_arr == arrR:
                for i in range(len(arrL)):
                    final_arr.append(arrL[i])

            elif arrL[i] > arrR[k] and arrR[k] > s:
                s = arrR[k]
                
                final_arr.append(s)
                
                if len(final_arr) + 1 == length:
                    final_arr.append(arrL[i])

    c = length - len(final_arr)  
    if c > 0:
        if arrR[len(arrR) - 1] > arrL[len(arrL) - 1]:
            for i in range(c):
                final_arr.append(arrR[len(arrR) - c + i])
        if arrR[len(arrR) - 1] < arrL[len(arrL) - 1]:
            for i in range(c):
                final_arr.append(arrL[len(arrL) - c + i])
            
            
        
    return final_arr
     

print(mergeSort(arr))