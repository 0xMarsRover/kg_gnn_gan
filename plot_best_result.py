# Plot results for paper
# pip install matplotlib==3.4.3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
'''
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
'''

labels = ['Text-only', 'Image-only', 'Text & Image']
# hmdb51, ucf101
hmdb51 = np.array([31.75, 31.41, 36.05])
ucf101 = np.array([29.09, 45.87, 46.37])

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, hmdb51, width, label='HMDB51', color='coral')
rects2 = ax.bar(x + width/2, ucf101, width, label='UCF101', color='skyblue')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean Average Accuracy (%)', fontweight='bold', fontsize=11)
ax.set_ylim(25, 53)

#ax.set_title('The best results when including single object')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

#ax.set_xlabel('Object Inclusion Approaches')
#plt.axhline(y=baseline, color='r', linestyle='-')
#plt.legend(["baseline", "obj1", "obj2", "obj3"])
ax.legend()
#ax.text(1, baseline, baseline, va='center', ha="left", color='r',transform=ax.get_yaxis_transform())

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3, )

fig.tight_layout()
#plt.show()
sns.despine(bottom=True)  # removes right and top axis lines
plt.savefig('best_results_each.eps', dpi=1000)

