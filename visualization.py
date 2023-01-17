
import numpy as np
import matplotlib.pyplot as plt


def exp_StyleDomainShift():

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

    # Results of Style-Domain-Shift experiment (modify term in evaluate_model.py):
    # Default_domain_shifts = 0.6
    domain_shifts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ADE_E25 = [0.0811, 0.0800, 0.0796, 0.0793, 0.0853, 0.1183, 0.1817, 0.2714]
    FDE_E25 = [0.1084, 0.1028, 0.0997, 0.1019, 0.1144, 0.1482, 0.2129, 0.3121]

    # Visualizations
    plt.figure(1)
    plt.plot(domain_shifts, cvpr_ADE, "-ob", label="Invariant + modular")
    plt.plot(domain_shifts, ADE_E25, "-or", label="OURS")
    plt.legend(loc="upper left")
    plt.xlim(0.0, 1.0)
    plt.xlabel('Style Domain Shifts')
    plt.ylim(0.0, 0.30)
    plt.ylabel('ADE')
    plt.show()

    plt.figure(2)
    plt.plot(domain_shifts, cvpr_FDE, "-ob", label="Invariant + modular")
    plt.plot(domain_shifts, FDE_E25, "-or", label="OURS")
    plt.legend(loc="upper left")
    plt.xlim(0.0, 1.0)
    plt.xlabel('Style Domain Shifts')
    plt.ylim(0.0, 0.35)
    plt.ylabel('FDE')
    plt.show()


def exp_DomianAdaptation_1():

    # Exp1-Update (E25_exp1_Bn): Prior S (Weights of Gaussian), Posterior S, Decoders (input reconstruction, future prediction)
    # Fine-Tuning the trained model with different number of test batches 1:8 for 100 epochs.
    # and compare the results with no fine-tuning.

    # ####### Our approach to test data split for fine tuning ####### #
    # finetune_ratio = 0.1 is equivalent to 8 batches
    # [0.1, 0.7/8, 0.6/8, 0.5/8, 0.4/8, 0.3/8, 0.2/8, 0.1/8]
    # [0.1, 0.0875, 0.075, 0.0625, 0.05, 0.0375, 0.025, 0.0125]

    # ####### CVPR approach to test data split for fine tuning ####### #
    # reduceall: cnt*64
    # cnt = [1, 2, 3, 4, 5, 6, 7, 8]

    # Experiment name: E25 (No fine-tuning)
    E25_ADE = 0.1182
    E25_FDE = 0.1480

    # Experiment name: E25_exp1_B1 (fine-tuning of 1 batch: reduceall=1*64)
    E25_B1_ADE = 0.1015
    E25_B1_FDE = 0.1314

    # Experiment name: E25_exp1_B2 (fine-tuning of 2 batch: reduceall=2*64)
    E25_B2_ADE = 0.0903
    E25_B2_FDE = 0.1227

    # Experiment name: E25_exp1_B3 (fine-tuning of 3 batch: reduceall=3*64)
    E25_B3_ADE = 0.0873
    E25_B3_FDE = 0.1187

    # Experiment name: E25_exp1_B4 (fine-tuning of 4 batch: reduceall=4*64)
    E25_B4_ADE = 0.0857
    E25_B4_FDE = 0.1168

    # Experiment name: E25_exp1_B5 (fine-tuning of 5 batch: reduceall=5*64)
    E25_B5_ADE = 0.0832
    E25_B5_FDE = 0.1134

    # Experiment name: E25_exp1_B6 (fine-tuning of 6 batch: reduceall=6*64)
    E25_B6_ADE = 0.0844
    E25_B6_FDE = 0.1147

    # Experiment name: E25_exp1_B7 (fine-tuning of 7 batch: reduceall=7*64)
    E25_B7_ADE = 0.0840
    E25_B7_FDE = 0.1164

    # Experiment name: E25_exp1_B8 (fine-tuning of 8 batch: reduceall=8*64)
    E25_B8_ADE = 0.0831
    E25_B8_FDE = 0.1136

    # output
    ADE_out = [E25_B8_ADE, E25_B7_ADE, E25_B6_ADE, E25_B5_ADE, E25_B4_ADE, E25_B3_ADE, E25_B2_ADE, E25_B1_ADE, E25_ADE]
    FDE_out = [E25_B8_FDE, E25_B7_FDE, E25_B6_FDE, E25_B5_FDE, E25_B4_FDE, E25_B3_FDE, E25_B2_FDE, E25_B1_FDE, E25_FDE]

    return ADE_out, FDE_out

def exp_DomianAdaptation_2():

    # Exp2-Update (E25_exp2_Bn): Prior S (Weights of Gaussian & Coupling layers), Posterior S, Decoders (input reconstruction, future prediction)

    # Experiment name: E25 (No fine-tuning)
    E25_ADE = 0.1182
    E25_FDE = 0.1480

    # Experiment name: E25_exp2_B1 (fine-tuning of 1 batch: reduceall=1*64)
    E25_B1_ADE = 0.0990
    E25_B1_FDE = 0.1285

    # Experiment name: E25_exp2_B2 (fine-tuning of 2 batch: reduceall=2*64)
    E25_B2_ADE = 0.0904
    E25_B2_FDE = 0.1206

    # Experiment name: E25_exp2_B3 (fine-tuning of 3 batch: reduceall=3*64)
    E25_B3_ADE = 0.0885
    E25_B3_FDE = 0.1222

    # Experiment name: E25_exp2_B4 (fine-tuning of 4 batch: reduceall=4*64)
    E25_B4_ADE = 0.0858
    E25_B4_FDE = 0.1160

    # Experiment name: E25_exp2_B5 (fine-tuning of 5 batch: reduceall=5*64)
    E25_B5_ADE = 0.0857
    E25_B5_FDE = 0.1178

    # Experiment name: E25_exp2_B6 (fine-tuning of 6 batch: reduceall=6*64)
    E25_B6_ADE = 0.0829
    E25_B6_FDE = 0.1149

    # Experiment name: E25_exp2_B7 (fine-tuning of 7 batch: reduceall=7*64)
    E25_B7_ADE = 0.0835
    E25_B7_FDE = 0.1152

    # Experiment name: E25_exp2_B8 (fine-tuning of 8 batch: reduceall=8*64)
    E25_B8_ADE = 0.0838
    E25_B8_FDE = 0.1192

    # output
    ADE_out = [E25_B8_ADE, E25_B7_ADE, E25_B6_ADE, E25_B5_ADE, E25_B4_ADE, E25_B3_ADE, E25_B2_ADE, E25_B1_ADE, E25_ADE]
    FDE_out = [E25_B8_FDE, E25_B7_FDE, E25_B6_FDE, E25_B5_FDE, E25_B4_FDE, E25_B3_FDE, E25_B2_FDE, E25_B1_FDE, E25_FDE]

    return ADE_out, FDE_out


def exp_DomianAdaptation_cvpr():

    # Experiment name: E25 (No fine-tuning)
    Ecvpr_ADE = 0.1258
    Ecvpr_FDE = 0.1546

    # Experiment name: E25_exp3_B1 (fine-tuning of 1 batch: reduceall=1*64)
    Ecvpr_B1_ADE = 0.1151
    Ecvpr_B1_FDE = 0.1410

    # Experiment name: E25_exp3_B2 (fine-tuning of 2 batch: reduceall=2*64)
    Ecvpr_B2_ADE = 0.1111
    Ecvpr_B2_FDE = 0.1377

    # Experiment name: E25_exp3_B3 (fine-tuning of 3 batch: reduceall=3*64)
    Ecvpr_B3_ADE = 0.1086
    Ecvpr_B3_FDE = 0.1359

    # Experiment name: E25_exp3_B4 (fine-tuning of 4 batch: reduceall=4*64)
    Ecvpr_B4_ADE = 0.1083
    Ecvpr_B4_FDE = 0.1347

    # Experiment name: E25_exp3_B5 (fine-tuning of 5 batch: reduceall=5*64)
    Ecvpr_B5_ADE = 0.1055
    Ecvpr_B5_FDE = 0.1340

    # Experiment name: E25_exp3_B6 (fine-tuning of 6 batch: reduceall=6*64)
    Ecvpr_B6_ADE = 0.1050
    Ecvpr_B6_FDE = 0.1328

    # Experiment name: E25_exp3_B7 (fine-tuning of 7 batch: reduceall=7*64)
    Ecvpr_B7_ADE = 0.1054
    Ecvpr_B7_FDE = 0.1328

    # Experiment name: E25_exp3_B8 (fine-tuning of 8 batch: reduceall=8*64)
    Ecvpr_B8_ADE = 0.1051
    Ecvpr_B8_FDE = 0.1335

    # output
    ADE_out = [Ecvpr_B8_ADE, Ecvpr_B7_ADE, Ecvpr_B6_ADE, Ecvpr_B5_ADE, Ecvpr_B4_ADE, Ecvpr_B3_ADE, Ecvpr_B2_ADE, Ecvpr_B1_ADE, Ecvpr_ADE]
    FDE_out = [Ecvpr_B8_FDE, Ecvpr_B7_FDE, Ecvpr_B6_FDE, Ecvpr_B5_FDE, Ecvpr_B4_FDE, Ecvpr_B3_FDE, Ecvpr_B2_FDE, Ecvpr_B1_FDE, Ecvpr_FDE]

    return ADE_out, FDE_out


# def exp_DomianAdaptation_4():
#
#     # Exp4-Update (E25_exp4_Bn): Prior S (Weights of Gaussian), Posterior S, Decoders (input reconstruction, future prediction)
#     # Fine-tune for 5 seeds plot mean abd var in a plot (number of experiments is 5x8=40)


def vis_DA():

    exp1_ADE, exp1_FDE = exp_DomianAdaptation_1()
    exp2_ADE, exp2_FDE = exp_DomianAdaptation_2()
    exp_cvpr_ADE, exp_cvpr_FDE = exp_DomianAdaptation_cvpr()

    batch = [8, 7, 6, 5, 4, 3, 2, 1, 0]

    # Visualizations
    plt.figure()
    plt.plot(batch, exp1_ADE, "-ob", label="exp1")
    plt.plot(batch, exp2_ADE, "-or", label="exp2")
    plt.plot(batch, exp_cvpr_ADE, "-og", label="Invariant+Modular")
    plt.legend(loc="lower left")
    plt.xlim(-1, 9)
    plt.xlabel('Number of Batches Used for Fine-Tuning')
    plt.ylim(0.07, 0.13)
    plt.ylabel('ADE')
    plt.show()

    plt.figure()
    plt.plot(batch, exp1_FDE, "-ob", label="exp1")
    plt.plot(batch, exp2_FDE, "-or", label="exp2")
    plt.plot(batch, exp_cvpr_FDE, "-og", label="Invariant+Modular")
    plt.legend(loc="lower left")
    plt.xlim(-1, 9)
    plt.xlabel('Number of Batches Used for Fine-Tuning')
    plt.ylim(0.10, 0.16)
    plt.ylabel('FDE')
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
    ADE_E25_S5 = 0.1280
    FDE_E25_S5 = 0.1537

    ADE_seeds = [ADE_E25_S1, ADE_E25_S2, ADE_E25_S3, ADE_E25_S4, ADE_E25_S5]
    FDE_seeds = [FDE_E25_S1, FDE_E25_S2, FDE_E25_S3, FDE_E25_S4, FDE_E25_S5]

    ave_ADE_over_seeds = np.mean(ADE_seeds)
    ave_FDE_over_seeds = np.mean(FDE_seeds)

    print('Mean ADE over seeds:', ave_ADE_over_seeds)
    print('Mean FDE over seeds:', ave_FDE_over_seeds)

    std_ADE_over_seeds = np.std(ADE_seeds)
    std_FDE_over_seeds = np.std(FDE_seeds)

    print('STD ADE over seeds:', std_ADE_over_seeds)
    print('STD FDE over seeds:', std_FDE_over_seeds)

def exp_AblationStudies():

    # Exp1: Only Z
    ADE_E26_z = 0.1404
    FDE_E26_z = 0.1818

    # Exp2: Only S
    ADE_E27_z = 0.2737
    FDE_E27_z = 0.3200

    # Exp3: No Coupling-Layers in S and Z Priors
    ADE_E28_NoCL = 0.1078
    FDE_E28_NoCL = 0.1351

    # Exp4: Num-Samples 10, 10
    ADE_E29_num_samp = 0.
    FDE_E29_num_samp = 0.


if __name__ == "__main__":

    # exp_StyleDomainShift()
    vis_DA()
    # exp_Identifiability()
    # exp_AblationStudies()