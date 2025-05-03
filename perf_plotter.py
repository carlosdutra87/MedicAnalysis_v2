import matplotlib.pyplot as plt

labels = ['draw()', 'load_image()', 'temporal_averaging()', 'mean_over_time()', 'mean()', 'clip()', 'astype()', 'imread()']
times = [52.7, 40.3, 22.1, 5.8, 5.8, 13.2, 2.7, 5.1]

plt.figure(figsize=(10,5))
plt.barh(labels, times, color='indianred')
plt.xlabel("Tempo Cumulativo (s)")
plt.title("Funções Críticas no Consumo de Tempo")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()
