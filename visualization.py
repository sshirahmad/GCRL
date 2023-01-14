
import numpy as np
import matplotlib.pyplot as plt


def exp_StyleDomainShift():

    domain_shifts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Experiment name: CVPR domain shift style
    # Results:
    cvpr_ADE = [0.0579, 0.0644, 0.0696, 0.0826, 0.0892, 0.1259, 0.1906, 0.2716]
    cvpr_FDE = [0.0841, 0.0877, 0.0915, 0.1002, 0.1178, 0.1546, 0.2110, 0.2884]

    # Experiment name: E23
    # Encoder and Decoder: MLP (similar to cvpr)
    # num-sample: 1, 10
    # Pre-training: None
    # Input to S: Distance between Pedestrians (only Past 8/20)
    # Input to Z: Absolute location of the pedestrians
    # Reconstruction: Absolute locations of the pedestrians
    # Future Trajectories Prediction: Relative locations of the pedestrians in relation to the starting point.
    # Results:
    ADE_E23 = [0.0977, 0.0960, 0.0970, 0.0998, 0.1116, 0.1371, 0.1890, 0.2637]
    FDE_E23 = [0.1409, 0.1342, 0.1306, 0.1329, 0.1491, 0.1798, 0.2355, 0.3192]

    # Experiment name: E24
    # E23 + Input to Z: [Absolute location of the pedestrians] cat [Distance between Pedestrians]
    # Results: No improvement observed

    # Experiment name: E25
    # E23 + Encoder of the S modified: The average calculation per batch was removed. generative model needs more sample
    # for training, but cvpr is a discriminative model, so it represents samples of a domain in a batch by a
    # representative member (Mean Vector) by applying mean aggregation function.
    # Results:
    ADE_E25 = [0.0814, 0.0799, 0.0798, 0.0793, 0.0850, 0.1182, 0.1826, 0.2710]
    FDE_E25 = [0.1084, 0.1028, 0.0999, 0.1019, 0.1143, 0.1480, 0.2128, 0.3116]

    # Visualizations
    plt.figure(1)
    plt.plot(domain_shifts, cvpr_ADE, "-ob", label="CVPR")
    plt.plot(domain_shifts, ADE_E25, "-or", label="OURS")
    plt.legend(loc="upper left")
    plt.xlim(0.0, 1.0)
    plt.xlabel('Style Domain Shifts')
    plt.ylim(0.0, 0.30)
    plt.ylabel('ADE')
    plt.show()

    plt.figure(2)
    plt.plot(domain_shifts, cvpr_FDE, "-ob", label="CVPR")
    plt.plot(domain_shifts, FDE_E25, "-or", label="OURS")
    plt.legend(loc="upper left")
    plt.xlim(0.0, 1.0)
    plt.xlabel('Style Domain Shifts')
    plt.ylim(0.0, 0.35)
    plt.ylabel('FDE')
    plt.show()


def exp_DomianAdaptation():

    # Fine-Tuning the trained model with different number of test batches 1:8
    # and compare the results with no fine-tuning.
    # finetune_ratio = 0.1 is equivalent to 8 batches
    # [0.1, 0.7/8, 0.6/8, 0.5/8, 0.4/8, 0.3/8, 0.2/8, 0.1/8]
    # [0.1, 0.0875, 0.075, 0.0625, 0.05, 0.0375, 0.025, 0.0125]

    # Experiment name: E25 (No fine-tuning)
    E25_ADE = 0.1182
    E25_FDE = 0.1480

    # Experiment name: E25_B1 (fine-tuning of 1 batch: finetune_ratio=0.1/8)
    E25_B1_ADE = 0.1000
    E25_B1_FDE = 0.1287

    # Experiment name: E25_B2 (fine-tuning of 2 batch: finetune_ratio=0.2/8)
    E25_B2_ADE = 0.0900
    E25_B2_FDE = 0.1197

    # Experiment name: E25_B3 (fine-tuning of 3 batch: finetune_ratio=0.3/8)
    E25_B3_ADE = 0.0867
    E25_B3_FDE = 0.1164

    # Experiment name: E25_B4 (fine-tuning of 4 batch: finetune_ratio=0.4/8)
    E25_B4_ADE = 0.0870
    E25_B4_FDE = 0.1185

    # Experiment name: E25_B5 (fine-tuning of 5 batch: finetune_ratio=0.5/8)
    E25_B5_ADE = 0.0850
    E25_B5_FDE = 0.1168

    # Experiment name: E25_B6 (fine-tuning of 6 batch: finetune_ratio=0.6/8)
    E25_B6_ADE = 0.0843
    E25_B6_FDE = 0.1179

    # Experiment name: E25_B7 (fine-tuning of 7 batch: finetune_ratio=0.7/8)
    E25_B7_ADE = 0.0848
    E25_B7_FDE = 0.1166

    # Experiment name: E25_B8 (fine-tuning of 8 batch: finetune_ratio=0.1)
    E25_B8_ADE = 0.0848
    E25_B8_FDE = 0.1211

    batch =  [0.1, 0.7 / 8, 0.6 / 8, 0.5 / 8, 0.4 / 8, 0.3 / 8, 0.2 / 8, 0.1 / 8, 0.0]
    ADE_ft = [E25_B8_ADE, E25_B7_ADE, E25_B6_ADE, E25_B5_ADE, E25_B4_ADE, E25_B3_ADE, E25_B2_ADE, E25_B1_ADE, E25_ADE]
    FDE_ft = [E25_B8_FDE, E25_B7_FDE, E25_B6_FDE, E25_B5_FDE, E25_B4_FDE, E25_B3_FDE, E25_B2_FDE, E25_B1_FDE, E25_FDE]


    # Visualizations
    plt.figure()
    plt.plot(batch, ADE_ft, "-ob", label="ADE")
    #plt.plot(batch, FDE_ft, "-or", label="FDE")
    #plt.legend(loc="upper right")
    plt.xlim(-0.01, 0.11)
    plt.xlabel('Batch Ratio Applied for Fine-Tuning')
    plt.ylim(0.07, 0.13)
    plt.ylabel('ADE')
    plt.show()


def exp_Identifiability():

    # Number of Epochs
    # E25 (seed = 1)
    epoch = 686 - 372  # (=314)
    ADE_E25_S1 = 0.1182
    FDE_E25_S1 = 0.1480

    # E25_S2 (Seed = 2)
    epoch = 736 - 422
    ADE_E25_S2 = 0.1221
    FDE_E25_S2 = 0.1623

    # E25_S3 (Seed = 3)
    epoch = 736 - 422
    ADE_E25_S3 = 0.1156
    FDE_E25_S3 = 0.1555

    # E25_S4 (Seed = 4)
    epoch = 736 - 422
    ADE_E25_S4 = 0.1171
    FDE_E25_S4 = 0.1466

    # E25_S5 (Seed = 5)
    epoch = 736 - 422
    ADE_E25_S5 = 0.
    FDE_E25_S5 = 0.

    ADE_seeds = [ADE_E25_S1, ADE_E25_S2, ADE_E25_S3, ADE_E25_S4]
    FDE_seeds = [FDE_E25_S1, FDE_E25_S2, FDE_E25_S3, FDE_E25_S4]

    ave_ADE_over_seeds = np.mean(ADE_seeds)
    ave_FDE_over_seeds = np.mean(FDE_seeds)

    print('Mean ADE over seeds:', ave_ADE_over_seeds)
    print('Mean FDE over seeds:', ave_FDE_over_seeds)

    std_ADE_over_seeds = np.std(ADE_seeds)
    std_FDE_over_seeds = np.std(FDE_seeds)

    print('STD ADE over seeds:', std_ADE_over_seeds)
    print('STD FDE over seeds:', std_FDE_over_seeds)


if __name__ == "__main__":

    # exp_StyleDomainShift()
    # exp_DomianAdaptation()
    exp_Identifiability()
