import matplotlib.pyplot as plt

def plot_figure(x_axis, y_axis, x_name, y_name):
    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Plot the data points with lines
    plt.plot(x_axis, y_axis, marker='o', linestyle='-')
    
    # Add a title and axis labels
    plt.title(f'{x_name} vs. {y_name}')
    plt.xlabel(f'{x_name}')
    plt.ylabel(f'{y_name} (s)')
    
    # Set the axis limits to start from zero
    plt.xlim(0, max(x_axis) * 1.1)  # Allow for 10% margin
    plt.ylim(0, max(y_axis) * 1.1)
    
    # Show grid lines
    plt.grid(True)
    
    # Save the figure as a PNG file
    plt.savefig(f'{x_name}_vs_{y_name}.png')
    
    # Display the figure
    plt.show()

if __name__ == "__main__":
    # Sample data: Batch sizes and corresponding latencies
    x_axis = [32, 64, 128, 256, 512]
    y_axis = [10, 8, 7, 6, 5]  # These latency values are fictional; replace with actual data

    # Call the function to plot and save the figure
    plot_figure(x_axis, y_axis, 'example', 'example')

