#-*- coding: utf-8 -*-

'''
generate normalized gaussian kernel 
'''
from math import exp

from pylab import *

def gen_kernel(sigma, step):

    def gaussian(sigma, x):
        return exp(- 0.5 * (x*x) / (sigma * sigma))

    max_x = sigma * 3
    kernel = [ gaussian(sigma, 0) ]
    i = 1
    while i*step <= max_x:
        kernel.append( gaussian(sigma, i * step) )
        i += 1
        
    mirror_kernel = kernel[1:]
    mirror_kernel.reverse()
    
    kernel = mirror_kernel + kernel
    
    s = sum(kernel)

    return [i/s for i in kernel]

if __name__ == "__main__":
    step = 1.0
    sigma_a = 1.0 # -1.0 -> 0 <- 1.0
    sigma_b = 2.0

    kernel_a = gen_kernel(sigma_a, step)
    kernel_b = gen_kernel(sigma_b, step)
    
    sz = len(kernel_a)
    coor_a = [ (i - sz/2) * step for i in xrange(sz)]
    
    # print kernel_a

    for i in xrange(sz):
        plot(coor_a[i], kernel_a[i], "*")
        

    sz = len(kernel_b)
    coor_b = [(i - sz/2) * step for i in xrange(sz)]
    
    
    for i in xrange(sz):
        plot(coor_b[i], kernel_b[i], "o")
        
    show()

    print kernel_a 
    print kernel_b
