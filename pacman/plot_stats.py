#!/usr/bin/env python

import matplotlib.pyplot as plt

f = open('stats', 'r')
it = []
q = []

for line in f:
    tmp = line.split(' ')
    it.append(int(tmp[0]))
    q.append(float(tmp[1]))

plt.plot(it, q)
plt.grid(True)
plt.show()
