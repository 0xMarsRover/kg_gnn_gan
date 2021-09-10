# Plot results for paper
# pip install matplotlib==3.4.3
import matplotlib.pyplot as plt
import numpy as np

labels = ['Replacing', 'Appending', 'Averaging']
obj1 = [43.69, 51.96, 51.00]
obj2 = [17.08, 31.33, 36.28]
obj3 = [42.88, 48.55, 50.56]
baseline = 47.06

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x-width, obj1, width, label='Replacing')
rects2 = ax.bar(x, obj2, width, label='Appending')
rects3 = ax.bar(x+width, obj3, width, label='Averaging')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean Average Accuracy in  %')
ax.set_ylim(10,68)
ax.set_title('The results when including single object')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Object Inclusion Approaches')

plt.axhline(y=baseline, color='r', linestyle='-')
plt.legend(["baseline", "obj1", "obj2", "obj3"])

ax.text(1, baseline, baseline, va='center', ha="left", color='r',transform=ax.get_yaxis_transform())

ax.bar_label(rects1, padding=2)
ax.bar_label(rects2, padding=2)
ax.bar_label(rects3, padding=2)

#fig.tight_layout()
#plt.show()
plt.savefig('res1.png', dpi=600)