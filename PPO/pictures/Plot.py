import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

def main():

    # Average rewards

    df_candy_6 = pd.read_csv('..\\csvs\\run-Sep15_17-04-25_dc95b00f4f53_8_6_PPO-tag-Var_AvgReward_8_6_PPO.csv', sep=',',usecols=["Value"])
    df_candy_6["field size = 7"] = pd.read_csv('..\\csvs\\run-Sep15_17-06-51_dc95b00f4f53_7_6_PPO-tag-Var_AvgReward_7_6_PPO.csv', sep=',',usecols=["Value"])
    df_candy_6["field size = 6"] = pd.read_csv('..\\csvs\\run-Sep15_17-12-08_dc95b00f4f53_6_6_PPO-tag-Var_AvgReward_6_6_PPO.csv', sep=',',usecols=["Value"])
    df_candy_6["field size = 5"] = pd.read_csv('..\\csvs\\run-Sep15_17-27-19_dc95b00f4f53_5_6_PPO-tag-Var_AvgReward_5_6_PPO.csv', sep=',',usecols=["Value"])

    df_candy_5 = pd.read_csv('..\\csvs\\run-Sep15_17-31-50_dc95b00f4f53_8_5_PPO-tag-Var_AvgReward_8_5_PPO.csv', sep=',',usecols=["Value"])
    df_candy_5["field size = 7"] = pd.read_csv('..\\csvs\\run-Sep15_17-34-21_dc95b00f4f53_7_5_PPO-tag-Var_AvgReward_7_5_PPO.csv', sep=',',usecols=["Value"])
    df_candy_5["field size = 6"] = pd.read_csv('..\\csvs\\run-Sep15_17-43-52_dc95b00f4f53_6_5_PPO-tag-Var_AvgReward_6_5_PPO.csv', sep=',',usecols=["Value"])
    df_candy_5["field size = 5"] = pd.read_csv('..\\csvs\\run-Sep15_19-05-29_9649fb04c717_5_5_PPO-tag-Var_AvgReward_5_5_PPO.csv', sep=',',usecols=["Value"])

    df_candy_4 = pd.read_csv('..\\csvs\\run-Sep15_18-13-26_dc95b00f4f53_8_4_PPO-tag-Var_AvgReward_8_4_PPO.csv', sep=',',usecols=["Value"])
    df_candy_4["field size = 7"] = pd.read_csv('..\\csvs\\run-Sep15_18-16-35_dc95b00f4f53_7_4_PPO-tag-Var_AvgReward_7_4_PPO.csv', sep=',',usecols=["Value"])
    df_candy_4["field size = 6"] = pd.read_csv('..\\csvs\\run-Sep15_18-19-48_dc95b00f4f53_6_4_PPO-tag-Var_AvgReward_6_4_PPO.csv', sep=',',usecols=["Value"])
    df_candy_4["field size = 5"] = pd.read_csv('..\\csvs\\run-Sep15_19-18-19_e515e37ada06_5_4_PPO-tag-Var_AvgReward_5_4_PPO.csv', sep=',',usecols=["Value"])

    x = np.arange(500)

    y,z = 1,0 # smoothing values

    fig, ax = plt.subplots(nrows=3, ncols=1,figsize = (5,6))

    avg_reward_smoothed = savgol_filter(df_candy_6["field size = 5"], y,  z)
    ax[2].plot(x, avg_reward_smoothed,  label="field size = 5")
    avg_reward_smoothed = savgol_filter(df_candy_6["field size = 6"], y,  z)
    ax[2].plot(x, avg_reward_smoothed,  label="field size = 6")
    avg_reward_smoothed = savgol_filter(df_candy_6["field size = 7"], y,  z)
    ax[2].plot(x, avg_reward_smoothed,  label="field size = 7")
    avg_reward_smoothed = savgol_filter(df_candy_6.iloc[:,0], y,  z)
    ax[2].plot(x, avg_reward_smoothed,  label="field size = 8")

    ax[2].set_title("Number of candies: 6")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Average Reward")
    ax[2].grid(True)
    ax[2].set_xticks(np.arange(0,501,50))

    avg_reward_smoothed = savgol_filter(df_candy_5["field size = 5"], y,  z)
    ax[1].plot(x, avg_reward_smoothed,  label="field size = 5")
    avg_reward_smoothed = savgol_filter(df_candy_5["field size = 6"], y,  z)
    ax[1].plot(x, avg_reward_smoothed,  label="field size = 6")
    avg_reward_smoothed = savgol_filter(df_candy_5["field size = 7"], y,  z)
    ax[1].plot(x, avg_reward_smoothed,  label="field size = 7")
    avg_reward_smoothed = savgol_filter(df_candy_5.iloc[:,0], y,  z)
    ax[1].plot(x, avg_reward_smoothed,  label="field size = 8")

    ax[1].set_title("Number of candies: 5")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Average Reward")
    ax[1].grid(True)
    ax[1].set_xticks(np.arange(0,501,50))

    avg_reward_smoothed = savgol_filter(df_candy_4["field size = 5"], y,  z)
    ax[0].plot(x, avg_reward_smoothed,  label="field size = 5")
    avg_reward_smoothed = savgol_filter(df_candy_4["field size = 6"], y,  z)
    ax[0].plot(x, avg_reward_smoothed,  label="field size = 6")
    avg_reward_smoothed = savgol_filter(df_candy_4["field size = 7"], y,  z)
    ax[0].plot(x, avg_reward_smoothed,  label="field size = 7")
    avg_reward_smoothed = savgol_filter(df_candy_4.iloc[:,0], y,  z)
    ax[0].plot(x, avg_reward_smoothed,  label="field size = 8")

    ax[0].set_title("Number of candies: 4")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Average Reward")
    ax[0].grid(True)
    ax[0].set_xticks(np.arange(0,501,50))
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.65),fancybox=True, shadow=True, ncol=2)


    fig.tight_layout(h_pad=1, w_pad=5)

    fig.set_size_inches(w=6, h=6)

    plt.savefig("trainingPlot_AvgReward.png")
    plt.show()

    ### Scores

    df_candy_6 = pd.read_csv('..\\csvs\\run-Sep15_17-04-25_dc95b00f4f53_8_6_PPO-tag-Var_AvgReward_8_6_PPO.csv', sep=',',usecols=["Value"])
    df_candy_6["field size = 7"] = pd.read_csv('..\\csvs\\run-Sep15_17-06-51_dc95b00f4f53_7_6_PPO-tag-Var_AvgReward_7_6_PPO.csv', sep=',',usecols=["Value"])
    df_candy_6["field size = 6"] = pd.read_csv('..\\csvs\\run-Sep15_17-12-08_dc95b00f4f53_6_6_PPO-tag-Var_AvgReward_6_6_PPO.csv', sep=',',usecols=["Value"])
    df_candy_6["field size = 5"] = pd.read_csv('..\\csvs\\run-Sep15_17-27-19_dc95b00f4f53_5_6_PPO-tag-Var_AvgReward_5_6_PPO.csv', sep=',',usecols=["Value"])

    df_candy_5 = pd.read_csv('..\\csvs\\run-Sep15_17-31-50_dc95b00f4f53_8_5_PPO-tag-Var_AvgReward_8_5_PPO.csv', sep=',',usecols=["Value"])
    df_candy_5["field size = 7"] = pd.read_csv('..\\csvs\\run-Sep15_17-34-21_dc95b00f4f53_7_5_PPO-tag-Var_AvgReward_7_5_PPO.csv', sep=',',usecols=["Value"])
    df_candy_5["field size = 6"] = pd.read_csv('..\\csvs\\run-Sep15_17-43-52_dc95b00f4f53_6_5_PPO-tag-Var_AvgReward_6_5_PPO.csv', sep=',',usecols=["Value"])
    df_candy_5["field size = 5"] = pd.read_csv('..\\csvs\\run-Sep15_19-05-29_9649fb04c717_5_5_PPO-tag-Var_AvgReward_5_5_PPO.csv', sep=',',usecols=["Value"])

    df_candy_4 = pd.read_csv('..\\csvs\\run-Sep15_18-13-26_dc95b00f4f53_8_4_PPO-tag-Var_AvgReward_8_4_PPO.csv', sep=',',usecols=["Value"])
    df_candy_4["field size = 7"] = pd.read_csv('..\\csvs\\run-Sep15_18-16-35_dc95b00f4f53_7_4_PPO-tag-Var_AvgReward_7_4_PPO.csv', sep=',',usecols=["Value"])
    df_candy_4["field size = 6"] = pd.read_csv('..\\csvs\\run-Sep15_18-19-48_dc95b00f4f53_6_4_PPO-tag-Var_AvgReward_6_4_PPO.csv', sep=',',usecols=["Value"])
    df_candy_4["field size = 5"] = pd.read_csv('..\\csvs\\run-Sep15_19-18-19_e515e37ada06_5_4_PPO-tag-Var_AvgReward_5_4_PPO.csv', sep=',',usecols=["Value"])

    x = np.arange(500)

    y,z = 1,0 # smoothing values

    fig, ax = plt.subplots(nrows=3, ncols=1,figsize = (5,6))

    avg_reward_smoothed = savgol_filter(df_candy_6["field size = 5"], y,  z)
    ax[2].plot(x, avg_reward_smoothed,  label="field size = 5")
    avg_reward_smoothed = savgol_filter(df_candy_6["field size = 6"], y,  z)
    ax[2].plot(x, avg_reward_smoothed,  label="field size = 6")
    avg_reward_smoothed = savgol_filter(df_candy_6["field size = 7"], y,  z)
    ax[2].plot(x, avg_reward_smoothed,  label="field size = 7")
    avg_reward_smoothed = savgol_filter(df_candy_6.iloc[:,0], y,  z)
    ax[2].plot(x, avg_reward_smoothed,  label="field size = 8")

    ax[2].set_title("Number of candies: 6")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Average Reward")
    ax[2].grid(True)
    ax[2].set_xticks(np.arange(0,501,50))

    avg_reward_smoothed = savgol_filter(df_candy_5["field size = 5"], y,  z)
    ax[1].plot(x, avg_reward_smoothed,  label="field size = 5")
    avg_reward_smoothed = savgol_filter(df_candy_5["field size = 6"], y,  z)
    ax[1].plot(x, avg_reward_smoothed,  label="field size = 6")
    avg_reward_smoothed = savgol_filter(df_candy_5["field size = 7"], y,  z)
    ax[1].plot(x, avg_reward_smoothed,  label="field size = 7")
    avg_reward_smoothed = savgol_filter(df_candy_5.iloc[:,0], y,  z)
    ax[1].plot(x, avg_reward_smoothed,  label="field size = 8")

    ax[1].set_title("Number of candies: 5")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Average Reward")
    ax[1].grid(True)
    ax[1].set_xticks(np.arange(0,501,50))

    avg_reward_smoothed = savgol_filter(df_candy_4["field size = 5"], y,  z)
    ax[0].plot(x, avg_reward_smoothed,  label="field size = 5")
    avg_reward_smoothed = savgol_filter(df_candy_4["field size = 6"], y,  z)
    ax[0].plot(x, avg_reward_smoothed,  label="field size = 6")
    avg_reward_smoothed = savgol_filter(df_candy_4["field size = 7"], y,  z)
    ax[0].plot(x, avg_reward_smoothed,  label="field size = 7")
    avg_reward_smoothed = savgol_filter(df_candy_4.iloc[:,0], y,  z)
    ax[0].plot(x, avg_reward_smoothed,  label="field size = 8")

    ax[0].set_title("Number of candies: 4")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Average Reward")
    ax[0].grid(True)
    ax[0].set_xticks(np.arange(0,501,50))
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.65),fancybox=True, shadow=True, ncol=2)


    fig.tight_layout(h_pad=1, w_pad=5)

    fig.set_size_inches(w=6, h=6)

    plt.savefig("trainingPlot_AvgReward.png")
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
