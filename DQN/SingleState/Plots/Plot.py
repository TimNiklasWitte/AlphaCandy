import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import scipy.signal

def main():
        
    base_path = "../TensorBoard_csv_files/run-.-tag-AvgReward"
 
    NUM_ROWS = 3
    fig, ax = plt.subplots(nrows=NUM_ROWS, ncols=1)

    for num_candy in range(4, 7):
    
        plot_idx = num_candy - 4
    
        for field_size in range(5, 9):

            file = f"{base_path}_{field_size}_{num_candy}_DQN.csv"

            data = pd.read_csv(file, sep=',')
            data = data["Value"]

            data = scipy.signal.savgol_filter(data, 101, 3, mode="mirror")
            ax[plot_idx].plot(data, label=f"field size = {field_size}")

            ax[plot_idx].set_title(f"Number of candys: {num_candy}")
            ax[plot_idx].set_xlabel("Epoch")
            ax[plot_idx].set_ylabel("Average Reward")
            ax[plot_idx].grid(True)

            ax[plot_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax[NUM_ROWS - 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.65),
        fancybox=True, shadow=True, ncol=2)
    
    fig.set_size_inches(w=6, h=6)
    

    plt.tight_layout()
    plt.savefig("./average_reward.png")
    plt.show()
  



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")