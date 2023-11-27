import matplotlib.pyplot as plt

example = {'0% to 10%': {'count': 5, 'correct': 3},
 '10% to 20%': {'count': 40, 'correct': 17},
 '20% to 30%': {'count': 2, 'correct': 1},
 '30% to 40%': {'count': 0, 'correct': 0},
 '40% to 50%': {'count': 34, 'correct': 16},
 '50% to 60%': {'count': 0, 'correct': 0},
 '60% to 70%': {'count': 0, 'correct': 0},
 '70% to 80%': {'count': 0, 'correct': 0},
 '80% to 90%': {'count': 0, 'correct': 0},
 '90% to 100%': {'count': 0, 'correct': 0}}

cmap = plt.get_cmap("Blues")

fig, ax = plt.subplots()

x = [(i/10) for i in range(len(example))]# + [1.0]
# x_ticks = [str(i/10) for i in range(len(example))] + [1.0]
y = [v.get('correct')/v.get('count') if v.get('count') > 0 else 0 for v in example.values()]# + [0]

n, bins, patches = ax.hist(x, len(x)-1, weights=y, edgecolor='black')
# plt.xticks(x, x_ticks)
ax.set_ylim([0, 1])
ax.set_xlim([0, 1])
ax.set_title('Title')
ax.set_ylabel('Accuracy Bin')
ax.set_xlabel('Confidence Bin')
ax.text(0.4, 0.9, 'Calibration Error:\n0.001', fontsize=14, color="blue", transform=ax.transAxes, va='top', ha='center')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle="--", color='black')
for h, p in zip(y, patches):
    plt.setp(p, 'facecolor', cmap(h))

plt.show()