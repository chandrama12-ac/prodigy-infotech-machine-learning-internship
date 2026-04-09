import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_data_generators

def compare_models(data_dir, model_names=['MobileNetV2', 'ResNet50']):
    """
    Compares multiple models on speed (inference time) and accuracy.
    """
    _, val_gen = get_data_generators(data_dir, batch_size=1, target_size=(224, 224))
    
    results = {}

    for name in model_names:
        model_path = f"models/{name}_best.keras"
        if not os.path.exists(model_path):
            print(f"Skipping {name}: Model file not found.")
            continue
            
        print(f"Evaluating {name}...")
        model = tf.keras.models.load_model(model_path)
        
        # Benchmarking Speed
        start_time = time.time()
        # Predict on 100 samples for average
        for i in range(100):
            img, _ = next(val_gen)
            model.predict(img, verbose=0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Benchmarking Accuracy
        loss, acc = model.evaluate(val_gen, steps=len(val_gen), verbose=0)
        
        results[name] = {
            "Accuracy": acc,
            "Latency (ms)": avg_time * 1000
        }
    
    # Visualization
    if results:
        names = list(results.keys())
        accs = [r['Accuracy'] for r in results.values()]
        times = [r['Latency (ms)'] for r in results.values()]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.bar(names, accs, color=color, alpha=0.6, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Latency (ms)', color=color)
        ax2.plot(names, times, color=color, marker='o', linewidth=2, label='Latency')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Performance Comparison: MobileNetV2 vs ResNet50')
        plt.show()
        
        print("\nFinal Comparison Table:")
        print(f"{'Model':<15} | {'Accuracy':<10} | {'Latency (ms)':<15}")
        print("-" * 45)
        for name, metrics in results.items():
            print(f"{name:<15} | {metrics['Accuracy']:.2%} | {metrics['Latency (ms)']:>12.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/food-101/images")
    args = parser.parse_args()
    
    compare_models(args.data_dir)
