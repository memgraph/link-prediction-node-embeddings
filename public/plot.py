from matplotlib import pyplot as plt
import numpy as np

#tribute to https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars
with open('../results.txt', 'r') as fh:
    lines = fh.readlines()


parsed_results=[]
for line in lines:
    values = line.split(" ")
    parsed_list=[float(value) for value in values]
    parsed_results.append(parsed_list[:2**5])

stddev = np.std(parsed_results, axis=0)
mean = np.mean(parsed_results, axis=0)
print(parsed_results)
x = np.arange(1,len(parsed_results[0])+1)
y = mean
error = stddev
error_min = y-error
error_max = y+error
print(error_max,"\n", error_min)
plt.plot(x, y, 'k-')
plt.fill_between(x, y-error, y+error)
plt.show()
