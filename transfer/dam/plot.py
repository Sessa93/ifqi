import numpy as np
import re
import matplotlib.pyplot as plt

def read_perf(filename):
    
    perf_file = open(filename, 'r')
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
        y_std.append(np.std(y) / np.sqrt(np.shape(y)[0]))
        
    x = np.array(x)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    return x,y_mean,y_std

x,y_mean1,y_std1 = read_perf("perf_dam_transfer_0.txt")
x,y_mean2,y_std2 = read_perf("perf_dam_transfer_20_1.txt")
x,y_mean3,y_std3 = read_perf("perf_dam_transfer_20_2.txt")
x,y_mean4,y_std4 = read_perf("perf_dam_transfer_20_3.txt")

fig, ax = plt.subplots()
ax.errorbar(x, y_mean1, yerr=y_std1, fmt='ro-', label='No Transfer')
ax.errorbar(x, y_mean2, yerr=y_std2, fmt='bo-', label='30 Years Source (1)')
ax.errorbar(x, y_mean3, yerr=y_std3, fmt='go-', label='30 Years Source (2)')
ax.errorbar(x, y_mean4, yerr=y_std4, fmt='yo-', label='30 Years Source (3)')
#x.errorbar(x, y3, yerr=y_err3, fmt='go-', label='1000 Source samp.')
#ax.errorbar(x, y4, yerr=y_err4, fmt='co-', label='2000 Source samp.')
plt.title("Dam Control")
ax.grid()
plt.xlim([0,101])
#plt.ylim([-40,-10])
plt.xlabel('Target Samples')

legend = ax.legend(loc='lower right', shadow=False)

frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('small')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width

plt.savefig('WFQI_dam.eps', format='eps', dpi=1200)
plt.show()
