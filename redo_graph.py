import argparse
import json
import numpy as np
from matplotlib import pyplot as plt


def save_graph(out_path, static_recalls, dynamic_recalls, static_times, dynamic_times):
    plt.plot(static_recalls, static_times, label="RP-Forest")
    plt.plot(dynamic_recalls, dynamic_times, label="MQ-Forest")
    
    plt.yscale('log')
    
    plt.grid(True, alpha=0.3, which='both')
    plt.minorticks_on()
    
    plt.ylim(bottom=10**2)
    
    plt.xlabel("Recall")
    plt.ylabel("Queries per Second")
    
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Regenerate graph from experiment JSON output')
    parser.add_argument('input_json', type=str, help='Path to input JSON file')
    parser.add_argument('output_file', type=str, help='Path to output graph file')
    
    args = parser.parse_args()
    
    # Load JSON data
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    # Extract data arrays
    static_recalls = np.array(data['static_recalls'])
    dynamic_recalls = np.array(data['dynamic_recalls'])
    static_qps = np.array(data['static_qps'])
    dynamic_qps = np.array(data['dynamic_qps'])
    
    # Generate graph
    save_graph(args.output_file, static_recalls, dynamic_recalls, static_qps, dynamic_qps)
    
    print(f"Graph saved to {args.output_file}")


if __name__ == "__main__":
    main()
