import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


avg = [35.15, 36.05, 32.72, 33.37]
sum = [34.94, 35.73, 32.44, 33.27]
max = [35.41, 36.32, 32.88, 33.61]
min = [35.34, 36.08, 33.30, 33.95]

dataset = np.array([avg, sum, max, min]).transpose()
df = pd.DataFrame(dataset, columns=['Avgering', 'Summation', 'Maximum', 'Minimum'])

for n in range(1,df.columns.shape[0]+1):
    df.rename(columns={f"data{n}": f"Experiment {n}"}, inplace=True)

vals, names, xs = [], [], []

for i, col in enumerate(df.columns):
    vals.append(df[col].values)
    names.append(col)
    xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

    plt.boxplot(vals, labels=names)
    palette = ['r', 'g', 'b', 'y']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)
    plt.show()

##### Set style options here #####
sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"

boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')
flierprops = dict(marker='o', markersize=1, linestyle='none')
whiskerprops = dict(color='#00145A')
capprops = dict(color='#00145A')
medianprops = dict(linewidth=1.5, linestyle='-', color='red')  # colors median line

plt.boxplot(vals, labels=names, notch=False, boxprops=boxprops, whiskerprops=whiskerprops,
            capprops=capprops, flierprops=flierprops, medianprops=medianprops,
            showmeans=False)  # notch=True adds median notch
ngroup = len(vals)

palette = ['#FF2709', '#09FF10', '#0030D7', '#FA70B5']

for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)

#plt.xlabel("Fusion Methods", fontweight='normal', fontsize=12)
plt.ylabel("Mean Average Accuracy (%)", fontweight='bold', fontsize=11)

sns.despine(bottom=True)  # removes right and top axis lines
#plt.title("The results of Dual-GANs using different fusion methods in HMDB51")
#plt.show()
plt.savefig('dual_hmdb51.png', dpi=800)

