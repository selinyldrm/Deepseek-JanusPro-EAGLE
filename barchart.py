import matplotlib.pyplot as plt

models = ['LlamaGen-XL-2', 'Lumina-mGPT']
counts = [1300, 2900]

plt.figure(figsize=(5, 4))
plt.bar(models, counts)
plt.ylabel('Frequency')
plt.xlabel('Q(x) = 0.0')
plt.title('Model Confidence Comparison')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./histogram.png')
plt.show()