import numpy as np
import re
import matplotlib.pyplot as plt

perf_file = open('perf_dam.txt', 'r')
content = perf_file.read()

match = re.findall(r"\[[\d\s\.\,\-]+\]",content)

x = []
y_mean = []
y_std = []

for m in match:
    
    run = [float(el) for el in m[1:-1].split(", ")]
    x.append(run[0])
    y = np.array(run[1:])
    y_mean.append(np.mean(y))
    y_std.append(np.std(y))

x = np.array(x)
y_mean = np.array(y_mean)
y_std = np.array(y_std)

plt.figure()
plt.errorbar(x, y_mean, yerr= y_std, fmt='-o')
plt.title("Dam Control")
plt.show()