# Plot results for paper
import matplotlib.pyplot as plt
import numpy as np

labels = ['Replacing', 'Appending', 'Averaging']
obj1 = [-3.37, 4.90, 3.94]
obj2 = [-29.98, -15.73, -10.78]
obj3 = [-4.18, 1.49, 3.50]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, obj1, width, label='Replacing')
rects2 = ax.bar(x + width/2, obj2, width, label='Appending')
rects3 = ax.bar(x + width/2, obj3, width, label='Appending')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Comparison Accuracy in  %')
ax.set_title('The comparison results against the baseline including single object')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.show()