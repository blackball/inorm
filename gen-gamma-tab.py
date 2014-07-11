#-*- coding: utf-8 -*-

def gen_gamma(range, gamma):
    tab = []

    for i in xrange(range[0], range[1] + 1):
        tab.append( round(pow(i/255.0, gamma) * 255.0, 5))

    return tab

if __name__ == "__main__":
    range = [0, 255]
    gamma = 0.2
    tab = gen_gamma(range, gamma)
    
    fp = open("gamma-tab.txt", "w")

    fp.write(", ".join( [str(i)+"f" for i in tab]))

    fp.close()

    
