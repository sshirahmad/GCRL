
import numpy as np
import matplotlib.pyplot as plt


def main():

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



if __name__ == "__main__":
    main()
