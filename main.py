import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv

# some global constants
Pop_vec = np.array([9.62 + 9.2 + 10.38 + 9.92,
                    10.99 + 10.46 + 11.02 + 10.54,
                    10.97 + 10.55 + 11.38 + 11.01,
                    11.67 + 11.43 + 11.26 + 11.04,
                    10.59 + 10.51 + 9.88 + 9.91 +
                    10.44 + 10.47 + 10.63 + 10.94,
                    10.33 + 10.9 + 8.75 + 9.65,
                    7.12 + 8.15 + 4.47 + 5.43 +
                    2.7 + 3.61 + 2.18 + 3.8], dtype=np.float64)
R0 = 5.7
beta = (R0 * (1 / 14.0)) / 0.4
M = np.array([[19.2, 4.8, 3, 7.1, 3.3943, 2.3, 1.4],
              [4.8, 42.4, 6.4, 5.4, 6.2262, 1.8, 1.7],
              [3, 6.4, 20.7, 9.2, 6.6924, 2, 0.9],
              [7.1, 5.4, 9.2, 16.9, 8.4185, 3.4, 1.5],
              [3.3943, 6.2262, 6.6924, 8.4185, 9.55, 3.0586, 1.9471],
              [2.3, 1.8, 2, 3.4, 3.0586, 7.5, 3.2],
              [1.4, 1.7, 0.9, 1.5, 1.9471, 3.2, 7.2]], dtype=np.float64)

# Asym rate (assume uniform distribution, incubation days = 4)
# n = 0.15
n = np.array([0.31, 0.362, 0.32, 0.28, 0.24, 0.18, 0.14])

# Symptomatic rate (incubation day = 4)
b = 1 - n
n /= 4.0
b /= 4.0

# Hospitalization rate vector (assume 14 day being infectious, each day has that prob of either hos or rec)
d = np.array([0.1, 0.3, 1.2, 3.2, 7.6, 16.6, 25.2842], dtype=np.float64) * 0.01

# Recover rate vector
c = (1 - d) / 14
d /= 142

# asym recover rate (14 day infection period)
alpha = np.ones(len(Pop_vec)) / 14.0

# Death rate (assume 4 days in hospital)
mu = np.array([0.002, 0.006, 0.03, 0.08, 0.1041, 2.2, 7.24], dtype=np.float64) * 0.01

# Hospitalization recover rate vector
r = (1 - mu) / 4
mu /= 4

# reinfection rate (grant 100 day immunity period) Gamma in paper
a = np.array([1 / 100.0 for i in c], dtype=np.float64)

# Suppose 10% of cases are identified(I), A class are not identified at all
percentage_identified = 0.014
percentage_Unidentified = 1 - percentage_identified


# -------------Protective measures--------------
M *= 0.16  # social-distancing originally = 0.12


# beta /= 1.5  # vaccination, add more recovered people at the beginning, reduce susceptible

# vaccination: 10% of population are vaccinated = 10% recovered.


def model(S, E, I, A, R, H, D, I_c, Age_grp, t_step, num_days):
    # Create a copy of each population compartment
    # Need to update each age compartment using params at that time
    S_fixed = S.copy()
    E_fixed = E.copy()
    I_fixed = I.copy()
    A_fixed = A.copy()
    R_fixed = R.copy()
    H_fixed = H.copy()

    for i in range(Age_grp):
        sum_coefficient = 0
        for j in range(Age_grp):
            sum_coefficient += M[i][j] * (I_fixed[j] + A_fixed[j]) * (1 - id(num_days)[j])

        dSdt = float(- beta * (S_fixed[i] / sum(Pop_vec)) * sum_coefficient + a[i] * R_fixed[i])

        dEdt = float(beta * (S_fixed[i] / sum(Pop_vec)) * sum_coefficient - b[i] * E_fixed[i] - n[i] * E_fixed[i])

        dI_cdt = b[i] * E_fixed[i] + n[i] * E_fixed[i]

        dIdt = (b[i] * E_fixed[i] - c[i] * I_fixed[i] - d[i] * I_fixed[i]) - id(num_days)[i] * I_fixed[i]

        dAdt = (n[i] * E_fixed[i] - alpha[i] * A_fixed[i]) - id(num_days)[i] * A_fixed[i]

        dRdt = (c[i] * I_fixed[i] + alpha[i] * A_fixed[i] + r[i] * H_fixed[i] - a[i] * R_fixed[i])

        dHdt = d[i] * I_fixed[i] - mu[i] * H_fixed[i] - r[i] * H_fixed[i]

        dDdt = mu[i] * H_fixed[i]

        # Euler's method updates
        S[i] += dSdt * t_step if S[i] + dSdt * t_step >= 0 else -S[i]
        E[i] += dEdt * t_step if E[i] + dEdt * t_step >= 0 else -E[i]
        I_c[i] += dI_cdt if dI_cdt * t_step >= 0 else -I_c[i]
        I[i] += dIdt * t_step if I[i] + dIdt * t_step >= 0 else -I[i]
        A[i] += dAdt * t_step if A[i] + dAdt * t_step >= 0 else -A[i]
        R[i] += dRdt * t_step if R[i] + dRdt * t_step >= 0 else -R[i]
        H[i] += dHdt * t_step if H[i] + dHdt * t_step >= 0 else -H[i]
        D[i] = dDdt * t_step

    return S, E, I, A, R, H, D, I_c


def get_rolling_data():
    with open('national-history.csv') as file:
        csvFile = csv.reader(file)
        raw_data = []
        for line in csvFile:
            raw_data.append(line[12])
        raw_data.pop(0)
        raw_data.pop(0)
        raw_data = np.array(raw_data, dtype='float64') * (1 / 1000000.0)

    rolling_data = []
    # construct the first 7
    for i in range(7):
        rolling_data.append(raw_data[i])

    # calculate rolling average based on prev 7
    for i in range(7, len(raw_data)):
        rolling_data.append((sum(raw_data[i - 6: i - 1]) + raw_data[i]) / 7.0)

    return rolling_data


def get_data():
    with open('Raw_DATA.csv', mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        case_by_week = []
        for lines in csvFile:
            case_by_week.append(lines[2])

        # manually overwrite 1st element as 0, and pop last few
        case_by_week[0] = 1
        case_by_week.pop(-1)
        case_by_week.pop(-1)
        for i in range(len(case_by_week)):
            case_by_week[i] = int(case_by_week[i]) / 1000000

        return case_by_week


# construct some bimodal function: first modal at t = 200, second modal at t = 300
def id(t):
    id_ = np.array([0.2, 0.18, 0.18, 0.16, 0.17, 0.18, 0.2], dtype=np.float64) * 0.02
    episolon = 0.000012
    return id_ + (episolon * t + 5 * episolon * (t-200) ** 2) / 100


def main():
    Rolling_data = get_rolling_data()

    Real_data = get_data()
    print(len(Rolling_data), len(Real_data))
    t_step = 0.5
    t_max = 350 # the simulation length
    R = Pop_vec / 5  # vaccination params
    S = Pop_vec.copy()  # all people are susceptible
    R = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  #this is raw group simulation
    S = Pop_vec - R
    num_groups = len(S)  # age groups number

    H = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    D = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    E = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    I = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    I_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    A = np.array([0.0, 0.0, 0.000001, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    vec_lst = [S, E, I, A, R, H, D, I_c]

    # cumulative Infected case

    S_sum = [sum(S)]
    E_sum = [sum(E)]
    I_sum = [sum(I)]
    A_sum = [sum(A)]
    R_sum = [sum(R)]
    H_sum = [sum(H)]
    D_sum = [sum(D)]
    I_c_sum = [0]
    T_span = [1]
    S_compartment = {"0-9": [S[0]], "10-19": [S[1]], "20-29": [S[2]],
                     "30-39": [S[3]], "40-59": [S[4]], "60-69": [S[5]], "70+": [S[6]]}
    E_compartment = {"0-9": [E[0]], "10-19": [E[1]], "20-29": [E[2]],
                     "30-39": [E[3]], "40-59": [E[4]], "60-69": [E[5]], "70+": [E[6]]}
    I_compartment = {"0-9": [I[0]], "10-19": [I[1]], "20-29": [I[2]],
                     "30-39": [I[3]], "40-59": [I[4]], "60-69": [I[5]], "70+": [I[6]]}
    A_compartment = {"0-9": [A[0]], "10-19": [A[1]], "20-29": [A[2]],
                     "30-39": [A[3]], "40-59": [A[4]], "60-69": [A[5]], "70+": [A[6]]}
    R_compartment = {"0-9": [R[0]], "10-19": [R[1]], "20-29": [R[2]],
                     "30-39": [R[3]], "40-59": [R[4]], "60-69": [R[5]], "70+": [R[6]]}
    H_compartment = {"0-9": [H[0]], "10-19": [H[1]], "20-29": [H[2]],
                     "30-39": [H[3]], "40-59": [H[4]], "60-69": [H[5]], "70+": [H[6]]}
    D_compartment = {"0-9": [D[0]], "10-19": [D[1]], "20-29": [D[2]],
                     "30-39": [D[3]], "40-59": [D[4]], "60-69": [D[5]], "70+": [D[6]]}
    I_c_compartment = {"0-9": [I_c[0]], "10-19": [I_c[1]], "20-29": [I_c[2]],
                     "30-39": [I_c[3]], "40-59": [I_c[4]], "60-69": [I_c[5]], "70+": [I_c[6]]}
    compartment_lst = [S_compartment, E_compartment, I_compartment,
                       A_compartment, R_compartment, H_compartment, D_compartment, I_c_compartment]

    for i in range(1, int(t_max / t_step)):
        S, E, I, A, R, H, D, I_c = model(S, E, I, A, R, H, D, I_c, num_groups, t_step, i)

        compartment_index = 0
        for compartment in compartment_lst:
            index = 0
            for key in compartment.keys():
                compartment[key].append(vec_lst[compartment_index][index])
                index += 1
            compartment_index += 1
        S_sum.append(sum(S))
        E_sum.append(sum(E))
        I_sum.append(sum(I))
        A_sum.append(sum(A))
        R_sum.append(sum(R))
        H_sum.append(sum(H))
        D_sum.append(sum(D))
        I_c_sum.append(sum(I_c * id(i)))
        T_span.append(T_span[-1] + 1)

    print(len(I_c_sum), len(T_span))
    # construct cumulative infectious population
    # I_cumulative = [I_sum[0]]
    # for i in range(1, len(I_sum)):
    #     if i % 2 == 0:
    #         I_cumulative.append(I_cumulative[-1] + I_sum[i])
    #     else:
    #         I_cumulative.append(I_cumulative[-1])

    # I_cumulative = [i * percentage_identified for i in I_cumulative[:]] # change I_cumu calculation formmat

    X_ = np.linspace(T_span[0], T_span[-1] * t_step, 500)

    plt.figure()
    cubic_S = interp1d(T_span, S_sum, kind="cubic")
    cubic_E = interp1d(T_span, E_sum, kind="cubic")
    cubic_I = interp1d(T_span, I_sum, kind="cubic")
    cubic_A = interp1d(T_span, A_sum, kind="cubic")
    cubic_R = interp1d(T_span, R_sum, kind="cubic")
    cubic_H = interp1d(T_span, H_sum, kind="cubic")
    cubic_D = interp1d(T_span, D_sum, kind="cubic")
    cubic_I_c = interp1d(T_span, I_c_sum, kind="cubic")

    # Figure 1--Population dynamics
    S_ = cubic_S(X_)
    E_ = cubic_E(X_)
    I_ = cubic_I(X_)
    A_ = cubic_A(X_)
    R_ = cubic_R(X_)
    H_ = cubic_H(X_)
    D_ = cubic_D(X_)
    I_c_ = cubic_I_c(X_)
    plt.plot(X_, S_, label='S')
    plt.plot(X_, E_, label='E')
    plt.plot(X_, I_, label='I')
    plt.plot(X_, A_, label='A')
    plt.plot(X_, R_, label='R')
    plt.plot(X_, H_, label='H')
    plt.plot(X_, D_, label='D')
    #plt.plot(X_, I_c_, label='Cumulative Infectious')

    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Number of Population (in million)')
    plt.title('Population dynamics')

    title_lst = ['Susceptible', 'Incubation', 'Infectious(Symptomatic)',
                 'Infectious(Asymptomatic)', 'Recovered', 'Hospitalized', 'Death', 'Cumulative Infectious']

    # Fig2, with comparison to real data
    plt.figure()
    plt.plot(X_, I_c_, label='Identified Infectious')
    # plot the real data
    num_week = [7 * i for i in range(int(t_max / 7))]
    case_week = [Real_data[i] for i in range(int(t_max / 7))]

    plt.plot(num_week, case_week, label='Real Data', color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=5)
    plt.plot(range(t_max), Rolling_data[:t_max], label='Rolling Daily Data', color='blue', linestyle='dashed', linewidth=3,
             marker='x', markerfacecolor='green', markersize=5)
    # plt.plot(X_, I_, label='I')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Number of Population (in million)')
    plt.title('Population dynamics')

    for i in range(t_max):
        if np.abs(I_c_sum[i] - Rolling_data[i]) < 0.1:
            print(i)


    # Figure 2~9 -- every class by age groups:
    graph_index = 0
    for compartment in compartment_lst:
        plt.figure()
        index = 0
        for key in compartment.keys():
            cubic_ = interp1d(T_span, np.array(compartment[key]), kind="cubic")
            compartment_ = cubic_(X_)
            plt.plot(X_, compartment_, label=key)
            index += 1
        plt.legend()
        plt.title(f'{title_lst[graph_index]} by age groups')
        graph_index += 1
        plt.xlabel('Days')
        plt.ylabel('Number of Population (in million)')
    plt.show()
    print(np.array([0.2, 0.18, 0.18, 0.16, 0.17, 0.18, 0.2], dtype=np.float64) * 0.02)


if __name__ == '__main__':
    main()
