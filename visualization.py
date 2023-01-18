
import numpy as np
import matplotlib.pyplot as plt


def exp_StyleDomainShift():

    # Experiment name: IM (IM) domain shift style
    # Results:
    IM_ADE = [0.0579, 0.0644, 0.0696, 0.0826, 0.0892, 0.1259, 0.1906, 0.2716]
    IM_FDE = [0.0841, 0.0877, 0.0915, 0.1002, 0.1178, 0.1546, 0.2110, 0.2884]

    # Experiment name: E23
    # Encoder and Decoder: MLP (similar to IM)
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
    # for training, but IM is a discriminative model, so it represents samples of a domain in a batch by a
    # representative member (Mean Vector) by applying mean aggregation function.

    # Results of Style-Domain-Shift experiment (modify term in evaluate_model.py):
    # Default_domain_shifts = 0.6
    domain_shifts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    ADE_E25 = [0.0811, 0.0800, 0.0796, 0.0793, 0.0853, 0.1183, 0.1817, 0.2714]
    FDE_E25 = [0.1084, 0.1028, 0.0997, 0.1019, 0.1144, 0.1482, 0.2129, 0.3121]

    # Visualizations
    plt.figure(1)
    plt.plot(domain_shifts, IM_ADE, "-ob", label="Invariant + modular")
    plt.plot(domain_shifts, ADE_E25, "-or", label="OURS")
    plt.legend(loc="upper left")
    plt.xlim(0.0, 1.0)
    plt.xlabel('Style Domain Shifts')
    plt.ylim(0.0, 0.30)
    plt.ylabel('ADE')
    plt.show()

    plt.figure(2)
    plt.plot(domain_shifts, IM_FDE, "-ob", label="Invariant + modular")
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

    # ####### IM approach to test data split for fine tuning ####### #
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
    ADE_out = [E25_B6_ADE, E25_B5_ADE, E25_B4_ADE, E25_B3_ADE, E25_B2_ADE, E25_B1_ADE, E25_ADE]
    FDE_out = [E25_B6_FDE, E25_B5_FDE, E25_B4_FDE, E25_B3_FDE, E25_B2_FDE, E25_B1_FDE, E25_FDE]

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
    ADE_out = [E25_B6_ADE, E25_B5_ADE, E25_B4_ADE, E25_B3_ADE, E25_B2_ADE, E25_B1_ADE, E25_ADE]
    FDE_out = [E25_B6_FDE, E25_B5_FDE, E25_B4_FDE, E25_B3_FDE, E25_B2_FDE, E25_B1_FDE, E25_FDE]

    return ADE_out, FDE_out


def exp_DomianAdaptation_IM_seed72():

    # Experiment name: E25 (No fine-tuning)
    EIM_ADE = 0.1258
    EIM_FDE = 0.1546

    # Experiment name: E25_exp3_B1 (fine-tuning of 1 batch: reduceall=1*64)
    EIM_B1_ADE = 0.1151
    EIM_B1_FDE = 0.1410

    # Experiment name: E25_exp3_B2 (fine-tuning of 2 batch: reduceall=2*64)
    EIM_B2_ADE = 0.1111
    EIM_B2_FDE = 0.1377

    # Experiment name: E25_exp3_B3 (fine-tuning of 3 batch: reduceall=3*64)
    EIM_B3_ADE = 0.1086
    EIM_B3_FDE = 0.1359

    # Experiment name: E25_exp3_B4 (fine-tuning of 4 batch: reduceall=4*64)
    EIM_B4_ADE = 0.1083
    EIM_B4_FDE = 0.1347

    # Experiment name: E25_exp3_B5 (fine-tuning of 5 batch: reduceall=5*64)
    EIM_B5_ADE = 0.1055
    EIM_B5_FDE = 0.1340

    # Experiment name: E25_exp3_B6 (fine-tuning of 6 batch: reduceall=6*64)
    EIM_B6_ADE = 0.1050
    EIM_B6_FDE = 0.1328

    # Experiment name: E25_exp3_B7 (fine-tuning of 7 batch: reduceall=7*64)
    EIM_B7_ADE = 0.1054
    EIM_B7_FDE = 0.1328

    # Experiment name: E25_exp3_B8 (fine-tuning of 8 batch: reduceall=8*64)
    EIM_B8_ADE = 0.1051
    EIM_B8_FDE = 0.1335

    # output
    ADE_out = [EIM_B8_ADE, EIM_B7_ADE, EIM_B6_ADE, EIM_B5_ADE, EIM_B4_ADE, EIM_B3_ADE, EIM_B2_ADE, EIM_B1_ADE, EIM_ADE]
    FDE_out = [EIM_B8_FDE, EIM_B7_FDE, EIM_B6_FDE, EIM_B5_FDE, EIM_B4_FDE, EIM_B3_FDE, EIM_B2_FDE, EIM_B1_FDE, EIM_FDE]

    return ADE_out, FDE_out


def exp_DomianAdaptation_IM():

    # Experiment name: E25 (No fine-tuning)
    EIM_ADE = 0.1034
    EIM_FDE = 0.1240

    # Experiment name: E25_exp3_B1 (fine-tuning of 1 batch: reduceall=1*64)
    EIM_B1_ADE = 0.1151
    EIM_B1_FDE = 0.1410

    # Experiment name: E25_exp3_B2 (fine-tuning of 2 batch: reduceall=2*64)
    EIM_B2_ADE = 0.0931
    EIM_B2_FDE = 0.1157

    # Experiment name: E25_exp3_B3 (fine-tuning of 3 batch: reduceall=3*64)
    EIM_B3_ADE = 0.092
    EIM_B3_FDE = 0.1126

    # Experiment name: E25_exp3_B4 (fine-tuning of 4 batch: reduceall=4*64)
    EIM_B4_ADE = 0.0936
    EIM_B4_FDE = 0.1145

    # Experiment name: E25_exp3_B5 (fine-tuning of 5 batch: reduceall=5*64)
    EIM_B5_ADE = 0.0898
    EIM_B5_FDE = 0.1106

    # Experiment name: E25_exp3_B6 (fine-tuning of 6 batch: reduceall=6*64)
    EIM_B6_ADE = 0.0873
    EIM_B6_FDE = 0.1081

    # Experiment name: E25_exp3_B7 (fine-tuning of 7 batch: reduceall=7*64)
    EIM_B7_ADE = 0.1054
    EIM_B7_FDE = 0.1328

    # Experiment name: E25_exp3_B8 (fine-tuning of 8 batch: reduceall=8*64)
    EIM_B8_ADE = 0.1051
    EIM_B8_FDE = 0.1335

    # output
    ADE_out = [EIM_B6_ADE, EIM_B5_ADE, EIM_B4_ADE, EIM_B3_ADE, EIM_B2_ADE, EIM_B1_ADE, EIM_ADE]
    FDE_out = [EIM_B6_FDE, EIM_B5_FDE, EIM_B4_FDE, EIM_B3_FDE, EIM_B2_FDE, EIM_B1_FDE, EIM_FDE]

    return ADE_out, FDE_out


def vis_DA():

    exp1_ADE, exp1_FDE = exp_DomianAdaptation_1()
    exp2_ADE, exp2_FDE = exp_DomianAdaptation_2()
    exp_IM_ADE, exp_IM_FDE = exp_DomianAdaptation_IM()

    batch = [6, 5, 4, 3, 2, 1, 0]

    # Visualizations
    plt.figure()
    plt.plot(batch, exp1_ADE, "-ob", label="VCRL-E1")
    plt.plot(batch, exp2_ADE, "-or", label="VCRL-E2")
    plt.plot(batch, exp_IM_ADE, "-og", label="IM")
    plt.legend(loc="lower left")
    plt.xlim(-1, 9)
    plt.xlabel('Number of Batches Used for Fine-Tuning')
    plt.ylim(0.07, 0.13)
    plt.ylabel('ADE')
    plt.show()


def exp_Identifiability_VCRL():

    # Number of Epochs
    # ##### E25 (seed = 1) epoch = 686 - 372=314 ##### #
    ADE_E25_S1_ds1 = 0.0811
    FDE_E25_S1_ds1 = 0.1083

    ADE_E25_S1_ds2 = 0.0798
    FDE_E25_S1_ds2 = 0.1028

    ADE_E25_S1_ds3 = 0.0797
    FDE_E25_S1_ds3 = 0.0998

    ADE_E25_S1_ds4 = 0.0794
    FDE_E25_S1_ds4 = 0.1021

    ADE_E25_S1_ds5 = 0.0851
    FDE_E25_S1_ds5 = 0.1142

    ADE_E25_S1_ds6 = 0.1183
    FDE_E25_S1_ds6 = 0.1484

    ADE_E25_S1_ds7 = 0.1828
    FDE_E25_S1_ds7 = 0.2132

    ADE_E25_S1_ds8 = 0.2716
    FDE_E25_S1_ds8 = 0.3126

    # ##### E25_S2 (seed = 2) epoch=736-422=314 ##### #
    ADE_E25_S2_ds1 = 0.0832
    FDE_E25_S2_ds1 = 0.1077

    ADE_E25_S2_ds2 = 0.0841
    FDE_E25_S2_ds2 = 0.1067

    ADE_E25_S2_ds3 = 0.0863
    FDE_E25_S2_ds3 = 0.1083

    ADE_E25_S2_ds4 = 0.0876
    FDE_E25_S2_ds4 = 0.1151

    ADE_E25_S2_ds5 = 0.0934
    FDE_E25_S2_ds5 = 0.1308

    ADE_E25_S2_ds6 = 0.1246
    FDE_E25_S2_ds6 = 0.1672

    ADE_E25_S2_ds7 = 0.1873
    FDE_E25_S2_ds7 = 0.2325

    ADE_E25_S2_ds8 = 0.2743
    FDE_E25_S2_ds8 = 0.3312

    # ##### E25_S3 (seed = 3) epoch=736-422=314 ##### #
    ADE_E25_S3_ds1 = 0.0779
    FDE_E25_S3_ds1 = 0.1071

    ADE_E25_S3_ds2 = 0.0794
    FDE_E25_S3_ds2 = 0.1049

    ADE_E25_S3_ds3 = 0.0817
    FDE_E25_S3_ds3 = 0.1061

    ADE_E25_S3_ds4 = 0.0828
    FDE_E25_S3_ds4 = 0.1113

    ADE_E25_S3_ds5 = 0.0875
    FDE_E25_S3_ds5 = 0.1242

    ADE_E25_S3_ds6 = 0.1158
    FDE_E25_S3_ds6 = 0.1558

    ADE_E25_S3_ds7 = 0.1787
    FDE_E25_S3_ds7 = 0.2195

    ADE_E25_S3_ds8 = 0.2688
    FDE_E25_S3_ds8 = 0.3197

    # ##### E25_S4 (seed = 4) epoch=736-422=314 ##### #
    ADE_E25_S4_ds1 = 0.0754
    FDE_E25_S4_ds1 = 0.1037

    ADE_E25_S4_ds2 = 0.0740
    FDE_E25_S4_ds2 = 0.0973

    ADE_E25_S4_ds3 = 0.0754
    FDE_E25_S4_ds3 = 0.0950

    ADE_E25_S4_ds4 = 0.0779
    FDE_E25_S4_ds4 = 0.0994

    ADE_E25_S4_ds5 = 0.0866
    FDE_E25_S4_ds5 = 0.1141

    ADE_E25_S4_ds6 = 0.1174
    FDE_E25_S4_ds6 = 0.1467

    ADE_E25_S4_ds7 = 0.1771
    FDE_E25_S4_ds7 = 0.2069

    ADE_E25_S4_ds8 = 0.2617
    FDE_E25_S4_ds8 = 0.3002

    # ##### E25_S5 (seed = 5) epoch=736-422=314 ##### #
    ADE_E25_S5_ds1 = 0.0923
    FDE_E25_S5_ds1 = 0.1267

    ADE_E25_S5_ds2 = 0.0905
    FDE_E25_S5_ds2 = 0.1200

    ADE_E25_S5_ds3 = 0.0889
    FDE_E25_S5_ds3 = 0.1141

    ADE_E25_S5_ds4 = 0.0871
    FDE_E25_S5_ds4 = 0.1129

    ADE_E25_S5_ds5 = 0.0940
    FDE_E25_S5_ds5 = 0.1228

    ADE_E25_S5_ds6 = 0.1277
    FDE_E25_S5_ds6 = 0.1544

    ADE_E25_S5_ds7 = 0.1907
    FDE_E25_S5_ds7 = 0.2156

    ADE_E25_S5_ds8 = 0.2776
    FDE_E25_S5_ds8 = 0.3121

    ADE_seeds_ds1 = [ADE_E25_S1_ds1, ADE_E25_S2_ds1, ADE_E25_S3_ds1, ADE_E25_S4_ds1, ADE_E25_S5_ds1]
    FDE_seeds_ds1 = [FDE_E25_S1_ds1, FDE_E25_S2_ds1, FDE_E25_S3_ds1, FDE_E25_S4_ds1, FDE_E25_S5_ds1]
    m_ade1, m_fde1, s_ade1, s_fde1 = get_mean_std_over_seeds(ADE_seeds_ds1, FDE_seeds_ds1, ds=0.1, model='VCRL')

    ADE_seeds_ds2 = [ADE_E25_S1_ds2, ADE_E25_S2_ds2, ADE_E25_S3_ds2, ADE_E25_S4_ds2, ADE_E25_S5_ds2]
    FDE_seeds_ds2 = [FDE_E25_S1_ds2, FDE_E25_S2_ds2, FDE_E25_S3_ds2, FDE_E25_S4_ds2, FDE_E25_S5_ds2]
    m_ade2, m_fde2, s_ade2, s_fde2 = get_mean_std_over_seeds(ADE_seeds_ds2, FDE_seeds_ds2, ds=0.2, model='VCRL')

    ADE_seeds_ds3 = [ADE_E25_S1_ds3, ADE_E25_S2_ds3, ADE_E25_S3_ds3, ADE_E25_S4_ds3, ADE_E25_S5_ds3]
    FDE_seeds_ds3 = [FDE_E25_S1_ds3, FDE_E25_S2_ds3, FDE_E25_S3_ds3, FDE_E25_S4_ds3, FDE_E25_S5_ds3]
    m_ade3, m_fde3, s_ade3, s_fde3 = get_mean_std_over_seeds(ADE_seeds_ds3, FDE_seeds_ds3, ds=0.3, model='VCRL')

    ADE_seeds_ds4 = [ADE_E25_S1_ds4, ADE_E25_S2_ds4, ADE_E25_S3_ds4, ADE_E25_S4_ds4, ADE_E25_S5_ds4]
    FDE_seeds_ds4 = [FDE_E25_S1_ds4, FDE_E25_S2_ds4, FDE_E25_S3_ds4, FDE_E25_S4_ds4, FDE_E25_S5_ds4]
    m_ade4, m_fde4, s_ade4, s_fde4 = get_mean_std_over_seeds(ADE_seeds_ds4, FDE_seeds_ds4, ds=0.4, model='VCRL')

    ADE_seeds_ds5 = [ADE_E25_S1_ds5, ADE_E25_S2_ds5, ADE_E25_S3_ds5, ADE_E25_S4_ds5, ADE_E25_S5_ds5]
    FDE_seeds_ds5 = [FDE_E25_S1_ds5, FDE_E25_S2_ds5, FDE_E25_S3_ds5, FDE_E25_S4_ds5, FDE_E25_S5_ds5]
    m_ade5, m_fde5, s_ade5, s_fde5 = get_mean_std_over_seeds(ADE_seeds_ds5, FDE_seeds_ds5, ds=0.5, model='VCRL')

    ADE_seeds_ds6 = [ADE_E25_S1_ds6, ADE_E25_S2_ds6, ADE_E25_S3_ds6, ADE_E25_S4_ds6, ADE_E25_S5_ds6]
    FDE_seeds_ds6 = [FDE_E25_S1_ds6, FDE_E25_S2_ds6, FDE_E25_S3_ds6, FDE_E25_S4_ds6, FDE_E25_S5_ds6]
    m_ade6, m_fde6, s_ade6, s_fde6 = get_mean_std_over_seeds(ADE_seeds_ds6, FDE_seeds_ds6, ds=0.6, model='VCRL')

    ADE_seeds_ds7 = [ADE_E25_S1_ds7, ADE_E25_S2_ds7, ADE_E25_S3_ds7, ADE_E25_S4_ds7, ADE_E25_S5_ds7]
    FDE_seeds_ds7 = [FDE_E25_S1_ds7, FDE_E25_S2_ds7, FDE_E25_S3_ds7, FDE_E25_S4_ds7, FDE_E25_S5_ds7]
    m_ade7, m_fde7, s_ade7, s_fde7 = get_mean_std_over_seeds(ADE_seeds_ds7, FDE_seeds_ds7, ds=0.7, model='VCRL')

    ADE_seeds_ds8 = [ADE_E25_S1_ds8, ADE_E25_S2_ds8, ADE_E25_S3_ds8, ADE_E25_S4_ds8, ADE_E25_S5_ds8]
    FDE_seeds_ds8 = [FDE_E25_S1_ds8, FDE_E25_S2_ds8, FDE_E25_S3_ds8, FDE_E25_S4_ds8, FDE_E25_S5_ds8]
    m_ade8, m_fde8, s_ade8, s_fde8 = get_mean_std_over_seeds(ADE_seeds_ds8, FDE_seeds_ds8, ds=0.8, model='VCRL')

    means_ade = [m_ade1, m_ade2, m_ade3, m_ade4, m_ade5, m_ade6, m_ade7, m_ade8]
    stds_ade = [s_ade1, s_ade2, s_ade3, s_ade4, s_ade5, s_ade6, s_ade7, s_ade8]

    means_fde = [m_fde1, m_fde2, m_fde3, m_fde4, m_fde5, m_fde6, m_fde7, m_fde8]
    stds_fde = [s_fde1, s_fde2, s_fde3, s_fde4, s_fde5, s_fde6, s_fde7, s_fde8]

    return means_ade, means_fde, stds_ade, stds_fde


def exp_Identifiability_IM():

    # Number of Epochs
    # ##### EIM (seed = 1) epochs=470 ##### #
    ADE_EIM_S1_ds1 = 0.0498
    FDE_EIM_S1_ds1 = 0.0708

    ADE_EIM_S1_ds2 = 0.0576
    FDE_EIM_S1_ds2 = 0.0731

    ADE_EIM_S1_ds3 = 0.0611
    FDE_EIM_S1_ds3 = 0.0777

    ADE_EIM_S1_ds4 = 0.0686
    FDE_EIM_S1_ds4 = 0.0865

    ADE_EIM_S1_ds5 = 0.0806
    FDE_EIM_S1_ds5 = 0.0995

    ADE_EIM_S1_ds6 = 0.1056
    FDE_EIM_S1_ds6 = 0.1300

    ADE_EIM_S1_ds7 = 0.1522
    FDE_EIM_S1_ds7 = 0.1868

    ADE_EIM_S1_ds8 = 0.2174
    FDE_EIM_S1_ds8 = 0.2671

    # ##### EIM_S2 (seed = 2) epochs=470 ##### #
    ADE_EIM_S2_ds1 = 0.0484
    FDE_EIM_S2_ds1 = 0.0694

    ADE_EIM_S2_ds2 = 0.0534
    FDE_EIM_S2_ds2 = 0.0692

    ADE_EIM_S2_ds3 = 0.0575
    FDE_EIM_S2_ds3 = 0.0732

    ADE_EIM_S2_ds4 = 0.0671
    FDE_EIM_S2_ds4 = 0.0838

    ADE_EIM_S2_ds5 = 0.0794
    FDE_EIM_S2_ds5 = 0.0996

    ADE_EIM_S2_ds6 = 0.1090
    FDE_EIM_S2_ds6 = 0.1317

    ADE_EIM_S2_ds7 = 0.1578
    FDE_EIM_S2_ds7 = 0.1842

    ADE_EIM_S2_ds8 = 0.2253
    FDE_EIM_S2_ds8 = 0.2575

    # ##### EIM_S3 (seed = 3) epochs=470 ##### #
    ADE_EIM_S3_ds1 = 0.0515
    FDE_EIM_S3_ds1 = 0.0767

    ADE_EIM_S3_ds2 = 0.0550
    FDE_EIM_S3_ds2 = 0.0771

    ADE_EIM_S3_ds3 = 0.0582
    FDE_EIM_S3_ds3 = 0.0779

    ADE_EIM_S3_ds4 = 0.0696
    FDE_EIM_S3_ds4 = 0.0845

    ADE_EIM_S3_ds5 = 0.0752
    FDE_EIM_S3_ds5 = 0.0961

    ADE_EIM_S3_ds6 = 0.1034
    FDE_EIM_S3_ds6 = 0.1240

    ADE_EIM_S3_ds7 = 0.1494
    FDE_EIM_S3_ds7 = 0.1669

    ADE_EIM_S3_ds8 = 0.2091
    FDE_EIM_S3_ds8 = 0.2255

    # ##### EIM_S4 (seed = 4) epochs=470 ##### #
    ADE_EIM_S4_ds1 = 0.0486
    FDE_EIM_S4_ds1 = 0.0825

    ADE_EIM_S4_ds2 = 0.0536
    FDE_EIM_S4_ds2 = 0.0805

    ADE_EIM_S4_ds3 = 0.0572
    FDE_EIM_S4_ds3 = 0.0840

    ADE_EIM_S4_ds4 = 0.0689
    FDE_EIM_S4_ds4 = 0.0939

    ADE_EIM_S4_ds5 = 0.0768
    FDE_EIM_S4_ds5 = 0.1022

    ADE_EIM_S4_ds6 = 0.1001
    FDE_EIM_S4_ds6 = 0.1231

    ADE_EIM_S4_ds7 = 0.1460
    FDE_EIM_S4_ds7 = 0.1672

    ADE_EIM_S4_ds8 = 0.2107
    FDE_EIM_S4_ds8 = 0.2339

    # ##### EIM_S5 (seed = 5) epochs=470 ##### #
    ADE_EIM_S5_ds1 = 0.0490
    FDE_EIM_S5_ds1 = 0.0717

    ADE_EIM_S5_ds2 = 0.0594
    FDE_EIM_S5_ds2 = 0.0755

    ADE_EIM_S5_ds3 = 0.0624
    FDE_EIM_S5_ds3 = 0.0796

    ADE_EIM_S5_ds4 = 0.0696
    FDE_EIM_S5_ds4 = 0.0879

    ADE_EIM_S5_ds5 = 0.0799
    FDE_EIM_S5_ds5 = 0.1008

    ADE_EIM_S5_ds6 = 0.1094
    FDE_EIM_S5_ds6 = 0.1320

    ADE_EIM_S5_ds7 = 0.1670
    FDE_EIM_S5_ds7 = 0.1959

    ADE_EIM_S5_ds8 = 0.2383
    FDE_EIM_S5_ds8 = 0.2782

    ADE_seeds_ds1 = [ADE_EIM_S1_ds1, ADE_EIM_S2_ds1, ADE_EIM_S3_ds1, ADE_EIM_S4_ds1, ADE_EIM_S5_ds1]
    FDE_seeds_ds1 = [FDE_EIM_S1_ds1, FDE_EIM_S2_ds1, FDE_EIM_S3_ds1, FDE_EIM_S4_ds1, FDE_EIM_S5_ds1]
    m_ade1, m_fde1, s_ade1, s_fde1 = get_mean_std_over_seeds(ADE_seeds_ds1, FDE_seeds_ds1, ds=0.1, model='IM')

    ADE_seeds_ds2 = [ADE_EIM_S1_ds2, ADE_EIM_S2_ds2, ADE_EIM_S3_ds2, ADE_EIM_S4_ds2, ADE_EIM_S5_ds2]
    FDE_seeds_ds2 = [FDE_EIM_S1_ds2, FDE_EIM_S2_ds2, FDE_EIM_S3_ds2, FDE_EIM_S4_ds2, FDE_EIM_S5_ds2]
    m_ade2, m_fde2, s_ade2, s_fde2 = get_mean_std_over_seeds(ADE_seeds_ds2, FDE_seeds_ds2, ds=0.2, model='IM')

    ADE_seeds_ds3 = [ADE_EIM_S1_ds3, ADE_EIM_S2_ds3, ADE_EIM_S3_ds3, ADE_EIM_S4_ds3, ADE_EIM_S5_ds3]
    FDE_seeds_ds3 = [FDE_EIM_S1_ds3, FDE_EIM_S2_ds3, FDE_EIM_S3_ds3, FDE_EIM_S4_ds3, FDE_EIM_S5_ds3]
    m_ade3, m_fde3, s_ade3, s_fde3 = get_mean_std_over_seeds(ADE_seeds_ds3, FDE_seeds_ds3, ds=0.3, model='IM')

    ADE_seeds_ds4 = [ADE_EIM_S1_ds4, ADE_EIM_S2_ds4, ADE_EIM_S3_ds4, ADE_EIM_S4_ds4, ADE_EIM_S5_ds4]
    FDE_seeds_ds4 = [FDE_EIM_S1_ds4, FDE_EIM_S2_ds4, FDE_EIM_S3_ds4, FDE_EIM_S4_ds4, FDE_EIM_S5_ds4]
    m_ade4, m_fde4, s_ade4, s_fde4 = get_mean_std_over_seeds(ADE_seeds_ds4, FDE_seeds_ds4, ds=0.4, model='IM')

    ADE_seeds_ds5 = [ADE_EIM_S1_ds5, ADE_EIM_S2_ds5, ADE_EIM_S3_ds5, ADE_EIM_S4_ds5, ADE_EIM_S5_ds5]
    FDE_seeds_ds5 = [FDE_EIM_S1_ds5, FDE_EIM_S2_ds5, FDE_EIM_S3_ds5, FDE_EIM_S4_ds5, FDE_EIM_S5_ds5]
    m_ade5, m_fde5, s_ade5, s_fde5 = get_mean_std_over_seeds(ADE_seeds_ds5, FDE_seeds_ds5, ds=0.5, model='IM')

    ADE_seeds_ds6 = [ADE_EIM_S1_ds6, ADE_EIM_S2_ds6, ADE_EIM_S3_ds6, ADE_EIM_S4_ds6, ADE_EIM_S5_ds6]
    FDE_seeds_ds6 = [FDE_EIM_S1_ds6, FDE_EIM_S2_ds6, FDE_EIM_S3_ds6, FDE_EIM_S4_ds6, FDE_EIM_S5_ds6]
    m_ade6, m_fde6, s_ade6, s_fde6 = get_mean_std_over_seeds(ADE_seeds_ds6, FDE_seeds_ds6, ds=0.6, model='IM')

    ADE_seeds_ds7 = [ADE_EIM_S1_ds7, ADE_EIM_S2_ds7, ADE_EIM_S3_ds7, ADE_EIM_S4_ds7, ADE_EIM_S5_ds7]
    FDE_seeds_ds7 = [FDE_EIM_S1_ds7, FDE_EIM_S2_ds7, FDE_EIM_S3_ds7, FDE_EIM_S4_ds7, FDE_EIM_S5_ds7]
    m_ade7, m_fde7, s_ade7, s_fde7 = get_mean_std_over_seeds(ADE_seeds_ds7, FDE_seeds_ds7, ds=0.7, model='IM')

    ADE_seeds_ds8 = [ADE_EIM_S1_ds8, ADE_EIM_S2_ds8, ADE_EIM_S3_ds8, ADE_EIM_S4_ds8, ADE_EIM_S5_ds8]
    FDE_seeds_ds8 = [FDE_EIM_S1_ds8, FDE_EIM_S2_ds8, FDE_EIM_S3_ds8, FDE_EIM_S4_ds8, FDE_EIM_S5_ds8]
    m_ade8, m_fde8, s_ade8, s_fde8 = get_mean_std_over_seeds(ADE_seeds_ds8, FDE_seeds_ds8, ds=0.8, model='IM')

    means_ade = [m_ade1, m_ade2, m_ade3, m_ade4, m_ade5, m_ade6, m_ade7, m_ade8]
    stds_ade = [s_ade1, s_ade2, s_ade3, s_ade4, s_ade5, s_ade6, s_ade7, s_ade8]

    means_fde = [m_fde1, m_fde2, m_fde3, m_fde4, m_fde5, m_fde6, m_fde7, m_fde8]
    stds_fde = [s_fde1, s_fde2, s_fde3, s_fde4, s_fde5, s_fde6, s_fde7, s_fde8]

    return means_ade, means_fde, stds_ade, stds_fde


def get_mean_std_over_seeds(ADE_seeds, FDE_seeds, ds=0.6, model='VCRL'):

    ave_ADE_over_seeds = np.mean(ADE_seeds)
    ave_FDE_over_seeds = np.mean(FDE_seeds)

    print('******* MODEL is {}'.format(model))
    print(f'Mean ADE over seeds with test domain shift of {ds:g}:', ave_ADE_over_seeds)
    # print(f'Mean FDE over seeds with test domain shift of {ds:g}:', ave_FDE_over_seeds)

    std_ADE_over_seeds = np.std(ADE_seeds)
    std_FDE_over_seeds = np.std(FDE_seeds)

    print(f'STD ADE over seeds with test domain shift of {ds:g}:', std_ADE_over_seeds)
    #print(f'STD FDE over seeds with test domain shift of {ds:g}:', std_FDE_over_seeds)

    return ave_ADE_over_seeds, ave_FDE_over_seeds, std_ADE_over_seeds, std_FDE_over_seeds


def vis_mean_stds_seeds():

    m_ade_vcrl, m_fde_vcrl, s_ade_vcrl, s_fde_vcrl = exp_Identifiability_VCRL()
    m_ade_im, m_fde_im, s_ade_im, s_fde_im = exp_Identifiability_IM()

    domain_shifts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    plt.figure()
    plt.plot(domain_shifts, m_ade_vcrl, "o-g", label="VCRL")
    plt.fill_between(domain_shifts, np.array(m_ade_vcrl) - np.array(s_ade_vcrl), np.array(m_ade_vcrl) + np.array(s_ade_vcrl), alpha=.4, color='green')
    plt.plot(domain_shifts, m_ade_im, "o-r", label="IM")
    plt.fill_between(domain_shifts, np.array(m_ade_im) - np.array(s_ade_im), np.array(m_ade_im) + np.array(s_ade_im), alpha=.4, color='red')
    plt.xlabel("Domain Shifts", fontsize=15)
    plt.xlabel("ADE", fontsize=15)
    plt.legend(loc="upper left", fontsize=15)
    plt.show()


def exp_AblationStudies():

    # All Ablation studies are conducted with seed=1
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
    # vis_mean_stds_seeds()