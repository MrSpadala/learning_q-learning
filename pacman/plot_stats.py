#!/usr/bin/env python

import matplotlib.pyplot as plt

f = open('stats', 'r')
it_q = []
q = []
it_loss = []
loss = []

for line in f:
    tmp = line.split(' ')
    it_q.append(int(tmp[0]))
    q.append(float(tmp[1]))

f.close()
f = open('loss', 'r')
for line in f:
    tmp = line.split(' ')
    it_loss.append(int(tmp[0]))
    loss.append(float(tmp[1]))

f.close()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(it_q, q)
ax1.set_title('Mean max q-value')
ax2.plot(it_loss, loss)
ax2.set_title('Loss')
plt.show()


