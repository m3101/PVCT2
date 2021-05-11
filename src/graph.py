from matplotlib import pyplot as plt
plt.plot([0,1,2,3,4,5,6,7,8],[2**x for x in range(9)],c="r")
plt.title("Amélia Stress Levels")
plt.xlabel("Hours worked")
plt.ylabel("Whatever unit")
plt.legend(["Stress"])
plt.show()