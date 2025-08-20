import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

x = np.array([180, 90, 45, 24, 12, 6, 3])
y_eer = np.array([3.694, 3.922, 4.154, 5.191, 7.894, 13.68, 24.20])

plt.plot(x, y_eer, color='orange')             
plt.scatter(x, y_eer, color='black', s=30)  

# Add value labels above each point
for xi, yi in zip(x, y_eer):
    plt.text(xi, yi + 0.8, f"{yi:.2f}", ha='center', fontsize=10, fontname='Calibri')

plt.grid(True)              
plt.minorticks_on()         

plt.grid(which='major', linestyle='-', linewidth=0.8)           
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7)  

# Define x-ticks including 3 explicitly
xticks = list(np.arange(0, 190, 15))
if 3 not in xticks:
    xticks.append(3)
xticks.sort()
plt.xticks(xticks)

# Add red vertical dotted line at x=3
plt.axvline(x=3, color='red', linestyle=':', linewidth=2)

plt.title("CER for different amounts of target speaker data", fontweight='bold', fontname='Calibri', fontsize=16)
plt.xlabel("target speaker data duration (seconds)", fontname='Calibri', fontsize=14)
plt.ylabel("Character Error Rate (CER)", fontname='Calibri', fontsize=14)

plt.show()
