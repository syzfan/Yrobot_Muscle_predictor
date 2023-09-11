import os
import csv
import datetime
import math
import copy
import time
import numpy as np
import pandas as pd
import scipy
import scipy.io as scio
import matplotlib.pyplot as plt
import xlrd
import sys
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt ,savgol_filter
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import random
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
from sympy import diff
from sympy import symbols
from scipy import interpolate
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import ConcatDataset


def butter_bandpass(data, low_cut, high_cut, fs, order=4):
    """
    Butter worth band pass filter implementation
    :param data: The data need to be filtered
    :param low_cut: low pass frequency for filter
    :param high_cut: high pass frequency for filter
    :param fs: sampling frequency
    :param order: degree of order
    :return: Filtered data from butter worth filter
    """
    nyq = 0.5 * fs
    low = low_cut/nyq
    high = high_cut/nyq
    [b, a] = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def emg_filter(emgData, bandPassHighFrequency=400, bandPassLowFrequency=20, butterWorthLowPass=12,
               frequency=1000, numberOfOrder=4):
    """
    The filter designed for emg, with bandpass filter, rectify, then 4th low pass filter
    :param emgData: The raw emg data only pass one part each time
    :param bandPassHighFrequency: the upper bound for bandpass filter frequency
    :param bandPassLowFrequency: the lower bound for bandpass filter frequency
    :param butterWorthLowPass: the frequency for 4th butter worth filter frequency
    :param frequency: the frequency for raw emg data
    :param numberOfOrder: the order for low pass butter worth filter, in case we need to do changes
    :return: the filtered emg data with form of ndarray
    """
    [b, a] = butter(4, (2 * butterWorthLowPass) / frequency, 'lowpass', analog=False)
    emgDataFiltered = abs(butter_bandpass(emgData, bandPassLowFrequency,
                                          bandPassHighFrequency, frequency, numberOfOrder))
    emgDataFiltered = filtfilt(b, a, emgDataFiltered)
    return emgDataFiltered


def interpolate_EMG(Input_file):
    print("Start interpolate process")
    Input=pd.read_csv(Input_file)
    columns=Input.columns
    Time_Length = len(Input[columns[0]])
    for i in range(len(columns)):
        if("ACC" in columns[i] or "GYRO" in columns[i] or "IMU Time" in columns[i]):
            y_len = np.array(Input[columns[i]].dropna())
            x_len = np.linspace(1, len(y_len), len(y_len))
            f = interpolate.interp1d(x_len, y_len, kind="linear")
            xnew = np.linspace(1, len(y_len), Time_Length)
            ynew = f(xnew).tolist()
            Input[columns[i]] = ynew

    # for i in range(len(Input.columns)):
    #     column_names = Input.columns
    #     Input[column_names[i]] = MinMaxScaler(Input[column_names[i]])
    Input.to_csv(Input_file+"_interpolate.csv", index=False)
    print("interpolate process finished")
    return 0

def find_stride_num(dataframe):
    stride_num_list=[]
    for i in range(len(dataframe["EMG segPoint"])-1):
        if(len(stride_num_list)==0):
           if(dataframe["EMG segPoint"][i+1]>dataframe["EMG segPoint"][i]):
                stride_num_list.append([0,i])
        else:
           if (dataframe["EMG segPoint"][i + 1] > dataframe["EMG segPoint"][i]):
                stride_num_list.append([stride_num_list[-1][1]+1, i+1 ])
    return stride_num_list

def progress_bar(task_time):
    print("\r", end="")
    print("Data processing progress: {}%: ".format(task_time), "▋" * (task_time // 2), end="")
    sys.stdout.flush()

def save_to_pickle(save_data, save_number):
    save_file = 'e' + save_number + '.p'
    with open(save_file, 'wb') as fp:
        pickle.dump(save_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


def load_pickle_data(file_number):
    file_name = 'e' + file_number + '.p'
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    return data


def Value_max_in_stride(dataframe,stride_num_list):
    max={}
    for i in range(len(stride_num_list)):
        max[i]=(dataframe[stride_num_list[i][0]:stride_num_list[i][1]].max())
    return max

def Value_min_in_stride(dataframe,stride_num_list):
    min={}
    for i in range(len(stride_num_list)):
        min[i]=(dataframe[stride_num_list[i][0]:stride_num_list[i][1]].min())
    return min

def Catch_Exception_Out(File_name,Stride_num_list):
    dataframe=pd.read_csv(File_name)
    column_names = dataframe.columns
    mark = []
    for i in range(len(column_names)):
        if ("ACC" in str(column_names[i]) or "GYRO" in str(column_names[i])):
            max_EMG_list = Value_max_in_stride(dataframe[str(column_names[i])], Stride_num_list)
            min_EMG_list = Value_min_in_stride(dataframe[str(column_names[i])], Stride_num_list)
            All_Mean = Series(max_EMG_list).mean()
            ALL_Std = Series(max_EMG_list).std()
            All_Mean_2 = Series(min_EMG_list).mean()
            ALL_Std_2 = Series(min_EMG_list).std()
            upper = All_Mean + 3 * ALL_Std
            lower = All_Mean - 3 * ALL_Std
            upper_2 = All_Mean_2 + 3 * ALL_Std_2
            lower_2 = All_Mean_2 - 3 * ALL_Std_2
            for j in range(len(max_EMG_list)):
                if lower > max_EMG_list[j] or max_EMG_list[j] > upper:
                    mark.insert(0, j)
                if lower_2 > min_EMG_list[j] or min_EMG_list[j] > upper_2:
                    mark.insert(0, j)
            print(mark)
    mark = (sorted(set(sorted(mark, reverse=True)), reverse=True))
    print(mark)
    for m in range(len(mark)):
        dataframe.drop(dataframe.index[Stride_num_list[mark[m]][0] - 1:Stride_num_list[mark[m]][1]], inplace=True)
    dataframe.to_csv(File_name+"_Expout.csv")
    return 0


def Catch_EMG_Exception_Out(File_Name,Folder_Name,File_Tail,Stride_num_list):
    dataframe=pd.read_csv(File_Name)
    columns = dataframe.columns
    mark = []
    for i in range(len(columns)):
        if ("ACC" in str(columns[i]) or "GYRO" in str(columns[i]) or("Avanti" in str(columns[i]) and "EMG" in str(columns[i]))):
            max_EMG_list = Value_max_in_stride(dataframe[str(columns[i])], Stride_num_list)
            min_EMG_list = Value_min_in_stride(dataframe[str(columns[i])], Stride_num_list)
            All_Mean = Series(max_EMG_list).mean()
            ALL_Std = Series(max_EMG_list).std()
            All_Mean_2 = Series(min_EMG_list).mean()
            ALL_Std_2 = Series(min_EMG_list).std()
            upper = All_Mean + 3 * ALL_Std
            lower = All_Mean - 3 * ALL_Std
            upper_2 = All_Mean_2 + 3 * ALL_Std_2
            lower_2 = All_Mean_2 - 3 * ALL_Std_2
            for j in range(len(max_EMG_list)):
                if lower > max_EMG_list[j] or max_EMG_list[j] > upper:
                    mark.insert(0, j)
                if lower_2 > min_EMG_list[j] or min_EMG_list[j] > upper_2:
                    mark.insert(0, j)
            print(mark)
    mark = (sorted(set(sorted(mark, reverse=True)), reverse=True))
    print(mark)
    for m in range(len(mark)):
        dataframe.drop(dataframe.index[Stride_num_list[mark[m]][0] - 1:Stride_num_list[mark[m]][1]], inplace=True)

    if(not os.path.exists(Folder_Name+"/EMGexpout")):
        os.mkdir(Folder_Name+"/EMGexpout")
    dataframe.to_csv(Folder_Name+"/EMGexpout"+"\\"+File_Tail)
    return 0



def Interpolate_IMU(File_Name,Folder_Name,File_Tail):
    data=pd.read_csv(File_Name)
    Emg_endtime = 0
    Imu_endtime = 0
    Emg_data = {}
    column_names = data.columns
    Emg_data['EMG Time'] = []
    Emg_data['IMU Time'] = []

    for i in range(len(column_names)):
        if "Avanti" in str(column_names[i]):
            raw_timelist = data[column_names[i - 1]]
            raw_valuelist = data[column_names[i]]
            if Emg_endtime == 0 and "EMG" in str(column_names[i]):
                temp = raw_timelist.dropna()
                Emg_endtime = temp.iloc[-1]
                EMG_endpose = temp.index[-1]
                Valuelist = (raw_valuelist[0:EMG_endpose])
                Emg_data[str(column_names[i])] = np.array(Valuelist, dtype=float)
                data.drop(data.index[EMG_endpose:len(data)], inplace=True)
            elif Imu_endtime == 0 and "ACC" in str(column_names[i]):
                for j in range(len(raw_timelist)-1):
                        if(Emg_endtime>raw_timelist[j] and Emg_endtime<=raw_timelist[j+1]):
                            Imu_endtime = raw_timelist[j+1]
                            Imu_endpos = j+1
                            break
                if(Imu_endtime==0):
                    for j in range(len(raw_timelist)):
                        if math.isnan(raw_timelist[j]):
                            Imu_endtime = raw_timelist[j - 1]
                            Imu_endpos = j-1
                            break
    for i in range(len(column_names)):
        if "ACC" in str(column_names[i]) or "GYRO" in str(column_names[i]):
            print(column_names[i])
            y_len = np.array(data[column_names[i]])[0:Imu_endpos]
            x_len = np.linspace(1, len(y_len), len(y_len))
            f = interpolate.interp1d(x_len, y_len, kind="linear")
            xnew = np.linspace(1, len(y_len), EMG_endpose)
            ynew = f(xnew).tolist()
            data[column_names[i]] = ynew


    if(not os.path.exists(Folder_Name+"/Interpolate")):
        os.mkdir(Folder_Name+"/Interpolate")
    data.to_csv(Folder_Name+"/Interpolate"+"\\"+File_Tail)


def Findminmax_Index(File_name):
    data=pd.read_csv(File_name)
    columns=data.columns
    minmax_dict={}
    for i in range(len(columns)):
        if ("ACC" in str(columns[i]) or "GYRO" in str(columns[i])):
            minmax_dict[str(columns[i])]=[max(data[columns[i]]),min(data[columns[i]])]
    Slide_length = 30
    Test_data_RF = np.array(data["Avanti sensor 1: EMG 1"])
    Test_data_VL = np.array(data["Avanti sensor 2: EMG 2"])
    Test_data_BF = np.array(data["Avanti sensor 3: EMG 3"])
    Test_data_TA = np.array(data["Avanti sensor 4: EMG 4"])
    Test_data_GAS = np.array(data["Avanti sensor 5: EMG 5"])
    Test_data_SOL = np.array(data["Avanti sensor 6: EMG 6"])
    RF_Max = 0
    VL_Max = 0
    BF_Max = 0
    TA_Max = 0
    GAS_Max = 0
    SOL_Max = 0
    RF_Min = 0
    VL_Min = 0
    BF_Min = 0
    TA_Min = 0
    GAS_Min = 0
    SOL_Min = 0
    for i in range(len(Test_data_RF) - Slide_length):
        RF_Max = max(RF_Max, sum(Test_data_RF[i:i + Slide_length]))
    RF_Max = RF_Max / Slide_length
    for i in range(len(Test_data_VL) - Slide_length):
        VL_Max = max(VL_Max, sum(Test_data_VL[i:i + Slide_length]))
    VL_Max = VL_Max / Slide_length
    # return {"RF_MVC":RF_MVC,"VL_MVC":VL_MVC}
    for i in range(len(Test_data_BF) - Slide_length):
        BF_Max = max(BF_Max, sum(Test_data_BF[i:i + Slide_length]))
    BF_Max = BF_Max / Slide_length
    for i in range(len(Test_data_TA) - Slide_length):
        TA_Max = max(TA_Max, sum(Test_data_TA[i:i + Slide_length]))
    TA_Max = TA_Max / Slide_length
    for i in range(len(Test_data_GAS) - Slide_length):
        GAS_Max = max(GAS_Max, sum(Test_data_GAS[i:i + Slide_length]))
    GAS_Max = GAS_Max / Slide_length
    for i in range(len(Test_data_SOL) - Slide_length):
        SOL_Max = max(SOL_Max, sum(Test_data_SOL[i:i + Slide_length]))
    SOL_Max = SOL_Max / Slide_length

    Slide_length = 10
    for i in range(len(Test_data_RF) - Slide_length):
        RF_Min = min(RF_Min, sum(Test_data_RF[i:i + Slide_length]))
    RF_Min = RF_Min / Slide_length
    for i in range(len(Test_data_VL) - Slide_length):
        VL_Min = min(VL_Min, sum(Test_data_VL[i:i + Slide_length]))
    VL_Min = VL_Min / Slide_length
    # return {"RF_MVC":RF_MVC,"VL_MVC":VL_MVC}
    for i in range(len(Test_data_BF) - Slide_length):
        BF_Min = min(BF_Min, sum(Test_data_BF[i:i + Slide_length]))
    BF_Min = BF_Min / Slide_length
    for i in range(len(Test_data_TA) - Slide_length):
        TA_Min = min(TA_Min, sum(Test_data_TA[i:i + Slide_length]))
    TA_Min = TA_Min / Slide_length
    for i in range(len(Test_data_GAS) - Slide_length):
        GAS_Min = min(GAS_Min, sum(Test_data_GAS[i:i + Slide_length]))
    GAS_Min = GAS_Min / Slide_length
    for i in range(len(Test_data_SOL) - Slide_length):
        SOL_Min = min(SOL_Min, sum(Test_data_SOL[i:i + Slide_length]))
    SOL_Min = SOL_Min / Slide_length

    EMG_dict= {"Avanti sensor 1: EMG 1": [RF_Max, RF_Min], "Avanti sensor 2: EMG 2": [VL_Max, VL_Min],
            "Avanti sensor 3: EMG 3": [BF_Max, BF_Min], "Avanti sensor 4: EMG 4": [TA_Max, TA_Min],
            "Avanti sensor 5: EMG 5": [GAS_Max, GAS_Min], "Avanti sensor 6: EMG 6": [SOL_Max, SOL_Min]}
    for i,j in EMG_dict.items():
        minmax_dict[i]=j

    return minmax_dict

def Minmaxfile(File_name,minmax_dict):
    data = pd.read_csv(File_name)
    columns = data.columns
    for i in range(len(columns)):
        if ("ACC" in str(columns[i]) or "GYRO" in str(columns[i]) ):
            mx, mi = minmax_dict[str(columns[i])][0] , minmax_dict[str(columns[i])][1]
            data[columns[i]] = (data[columns[i]] - mi) / (mx - mi)
            while(1):
                if(data[columns[i]].max()>1):
                    data[columns[i]][data[columns[i]].idxmax()]=1
                else:
                    break
            while(1):
                if(data[columns[i]].min()<0):
                    data[columns[i]][data[columns[i]].idxmin()]=0
                else:
                    break
        if ("Avanti" in str(columns[i]) and "EMG" in str(columns[i])):
            mx, mi = minmax_dict[str(columns[i])][0] , minmax_dict[str(columns[i])][1]
            data[columns[i]] = (data[columns[i]] - mi) / (mx - mi)
            while(1):
                if(data[columns[i]].max()>1):
                    data[columns[i]][data[columns[i]].idxmax()]=1
                else:
                    data[columns[i]]=round(data[columns[i]]*100)
                    break
            while(1):
                if(data[columns[i]].min()<0):
                    data[columns[i]][data[columns[i]].idxmin()]=0
                else:
                    break
    data.to_csv(File_name+"_Minmax.csv")



def find_lower(prelist):
    heelstrike_point=[]
    heelstrike_value=[]
    flag = 0
    for i in range(len(prelist)):
        if (i > 200 and i < len(prelist) - 800):
            for j in range(800):
                if (prelist[i - j] < prelist[i]):
                    flag = 1
                    break
                if (prelist[i + j] < prelist[i]):
                    flag = 1
                    break
            if (flag == 0 and len(heelstrike_point)==0):
                heelstrike_point.append(i)
                heelstrike_value.append(prelist[i])
            elif (flag == 0 and (i-heelstrike_point[-1])>100):
                heelstrike_point.append(i)
                heelstrike_value.append(prelist[i])
        flag = 0
    # print(heelstrike_point)
    # print((heelstrike_point))

    # plt.scatter(tuple(heelstrike_point),heelstrike_value,color="orange")
    # plt.plot(range(len(prelist)),prelist)
    # plt.show()
    return heelstrike_point

def find_peak(prelist):
    print(prelist)
    heelstrike_point=[]
    heelstrike_value=[]
    flag = 0
    for i in range(len(prelist)):
        if (i > 200 and i < len(prelist) - 800):
            for j in range(800):
                if (prelist[i - j] > prelist[i]):
                    flag = 1
                    break
                if (prelist[i + j] > prelist[i]):
                    flag = 1
                    break
            if (flag == 0 and len(heelstrike_point)==0):
                heelstrike_point.append(i)
                heelstrike_value.append(prelist[i])
            elif (flag == 0 and (i-heelstrike_point[-1])>100):
                heelstrike_point.append(i)
                heelstrike_value.append(prelist[i])
        flag = 0
    # print(heelstrike_point)
    # print((heelstrike_point))

    # plt.scatter(tuple(heelstrike_point),heelstrike_value,color="orange")
    # plt.plot(range(len(prelist)),prelist)
    # plt.show()
    return heelstrike_point

def Find_heelstrike_in_IMU(Folder_Name,file_name,File_Tail="None",Out=0):
    data=pd.read_csv(file_name)
    columns=data.columns
    ACC_X,ACC_Y,ACC_Z=data["Avanti sensor 5: ACC.X 5"],data["Avanti sensor 5: ACC.Y 5"],data["Avanti sensor 5: ACC.Z 5"]
    GYRO_X,GYRO_Y,GYRO_Z=data["Avanti sensor 7: GYRO.X 7"],data["Avanti sensor 5: GYRO.Y 5"],data["Avanti sensor 5: GYRO.Z 5"]
    # Bertec_Segpoint=data["IMU segPoint"]
    Slide_window=20
    raw_list=GYRO_X.values.tolist()
    heel_strike_point=find_peak(raw_list)
    print(heel_strike_point)
    stride_list=[]




    zero_point=[]
    zero_value=[]
    percentage_list=[]
    # Bertec_point=[]
    # Bertec_value=[]

    for i in range(len(heel_strike_point)):
        for j in range(400):
            if(raw_list[heel_strike_point[i]+j]>0 and raw_list[heel_strike_point[i]+j+1]<0):
                zero_point.append(heel_strike_point[i]+j+15)
                zero_value.append(raw_list[heel_strike_point[i]+j+15])
                break
    first_step=zero_point[0]-(zero_point[1]-zero_point[0])
    if(first_step>0):
        zero_point.insert(0,first_step)
    else:
        zero_point.insert(0, 1)
    zero_value.insert(0,raw_list[zero_point[0]])
    start_frame=np.zeros(zero_point[0]-1).tolist()
    for j in range(zero_point[0]-1):
          stride_list.append(1)
    STRIDE_SPLIT = 99
    for i in range(len(zero_point)-1):
        for j in range(zero_point[i+1]-zero_point[i]):
            start_frame.append(round(j * STRIDE_SPLIT / (zero_point[i+1]-zero_point[i]) + 1))
    for i in range(len(raw_list)-zero_point[-1]+1):
        start_frame.append(0)


    for i in range(len(zero_point) - 1):

            for j in range(zero_point[i + 1] - zero_point[i]):
                stride_list.append(i + 1)
    for i in range(len(data)-len(stride_list)):
        stride_list.append(0)
    # for i in range(len(zero_point)-1):
    #     if(len(stride_list)==0):
    #          stride_list.append([zero_point[i]-1,zero_point[i+1]-1])
    #     else:
    #          stride_list.append([zero_point[i],zero_point[i+1]-1])
    if(Out==1):
        data.insert(0, "percentage", start_frame)
        data.insert(0, "stride", stride_list)
        if (not os.path.exists(Folder_Name + "/Percentage")):
            os.mkdir(Folder_Name + "/Percentage")
        data.to_csv(Folder_Name + "/Percentage" + "\\" + File_Tail)
        return 0
    stride_list = []
    for i in range(len(zero_point) - 1):
        if (len(stride_list) == 0):
            stride_list.append([zero_point[i] - 1, zero_point[i + 1] - 1])
        else:
            stride_list.append([zero_point[i], zero_point[i + 1] - 1])
    return stride_list


    # stride_list=find_stride_num(data)
    # for j in range(len(stride_list)):[0])
    #     Bertec_value.append(raw_list[stride_list[j][0]])

    # plt.scatter(tuple(zero_point),zero_value,color="blue",label="zero_point")
    # plt.plot(range(len(raw_list)),raw_list,color="orange")
    # # plt.scatter(tuple(Bertec_point),Bertec_value,color="orange",label="bertec point")
    # plt.legend()
    # plt.show()



def Preprocess_Delsys(File_name,Folder_name,File_Tail):

    data=pd.read_csv(File_name)
    columns=data.columns
    for i in range(len(data.columns)):
        if("EMG" in str(data[columns[i]])):
            data[columns[i]]=emg_filter(data[columns[i]])
    if(not os.path.exists(Folder_name+"\\Filtered")):
        os.mkdir(Folder_name+"\\Filtered")
    data.to_csv(Folder_name+"\\Filtered"+"\\"+File_Tail)





def UseDataLoader_Output_Emg(dataframe, seq_length):
    Data_len = len(dataframe)
    pos=0
    imu_list = []
    emg_list = []
    column_names = dataframe.columns
    for i in range(len(column_names)):
        if (("ACC" in str(column_names[i]) or "GYRO" in str(column_names[i]) )and
                (''.join([x for x in str(column_names[i].split(":")[0]) if x.isdigit()])) in {"1","4","7","8","9","10"}):
                imu_list.append(str(column_names[i]))

        if(("EMG" in str(column_names[i])) and
                (''.join([x for x in str(column_names[i].split(":")[0]) if x.isdigit()])) in {"1","2","3","4","5","6"}):
                emg_list.append(str(column_names[i]))
    # print(imu_list)
    # print(emg_list)
    output1 = np.zeros((Data_len, seq_length, len(imu_list)+len(emg_list)+1))

    for i in range(pos,Data_len-2*seq_length):
    # for i in range(500):
         for m in range(seq_length):
               ##########选定matrix的每一行，将input_dim=18个数据填进去
                temp_frame = dataframe.loc[i+m]
                real_data = []
                real_data2= []
                for k in range(len(imu_list)):
                    real_data.append(temp_frame[imu_list[k]])
                for p in range(len(emg_list)):
                    real_data.append(temp_frame[emg_list[p]])
                temp_frame1 = dataframe.loc[i + m + seq_length]
                real_data2=temp_frame1["percentage"]
                # print(np.append(np.array(real_data),np.array(real_data2)))
                output1[i][m] = np.append(np.array(real_data),np.array(real_data2))

         progress_bar(int(i*100/(Data_len-2*pos)))
    return output1

def UseDataLoader_Output_Emg_Without_Period(dataframe, seq_length):
    Data_len = len(dataframe)
    pos=0
    imu_list = []
    emg_list = []
    column_names = dataframe.columns
    for i in range(len(column_names)):
        if (("ACC" in str(column_names[i]) or "GYRO" in str(column_names[i]) )and
                (''.join([x for x in str(column_names[i].split(":")[0]) if x.isdigit()])) in {"1","4","7","8","9","10"}):
                imu_list.append(str(column_names[i]))

        if(("EMG" in str(column_names[i])) and
                (''.join([x for x in str(column_names[i].split(":")[0]) if x.isdigit()])) in {"1","2","3","4","5","6"}):
                emg_list.append(str(column_names[i]))
    # print(imu_list)
    # print(emg_list)
    output1 = np.zeros((Data_len, seq_length, len(imu_list)+len(emg_list)))

    for i in range(pos,Data_len-2*seq_length):
    # for i in range(500):
         for m in range(seq_length):
               ##########选定matrix的每一行，将input_dim=18个数据填进去
                temp_frame = dataframe.loc[i+m]
                real_data = []
                real_data2= []
                for k in range(len(imu_list)):
                    real_data.append(temp_frame[imu_list[k]])
                for p in range(len(emg_list)):
                    real_data.append(temp_frame[emg_list[p]])
                temp_frame1 = dataframe.loc[i + m + seq_length]

                # print(np.append(np.array(real_data),np.array(real_data2)))
                output1[i][m] = np.array(real_data)

         progress_bar(int(i*100/(Data_len-2*pos)))

    return output1


def FindMVC_in_File(File_name):
    data=pd.read_csv(File_name)
    Slide_length = 50
    if("RF" in File_name and "filter" in File_name):
        Test_data_RF=np.array(data["Avanti sensor 1: EMG 1"])
        Test_data_VL=np.array(data["Avanti sensor 2: EMG 2"])
        RF_MVC=0
        VL_MVC=0

        for i in range(len(Test_data_RF)-Slide_length):
            RF_MVC=max(RF_MVC,sum(Test_data_RF[i:i+Slide_length]))
        RF_MVC=RF_MVC/Slide_length
        for i in range(len(Test_data_VL)-Slide_length):
            VL_MVC=max(VL_MVC,sum(Test_data_VL[i:i+Slide_length]))
        VL_MVC=VL_MVC/Slide_length
        # return {"RF_MVC":RF_MVC,"VL_MVC":VL_MVC}
        return {"Avanti sensor 1: EMG 1": RF_MVC,"Avanti sensor 2: EMG 2": VL_MVC}
    elif("BF" in File_name and "filter"  in File_name):
        Test_data_BF=np.array(data["Avanti sensor 3: EMG 3"])
        BF_MVC=0
        for i in range(len(Test_data_BF)-Slide_length):
            BF_MVC=max(BF_MVC,sum(Test_data_BF[i:i+Slide_length]))
        BF_MVC=BF_MVC/Slide_length
        # return {"BF_MVC":BF_MVC}
        return {"Avanti sensor 3: EMG 3":BF_MVC}
    elif("TA" in File_name and "filter"  in File_name):
        Test_data_TA=np.array(data["Avanti sensor 4: EMG 4"])
        TA_MVC = 0
        for i in range(len(Test_data_TA) - Slide_length):
            TA_MVC = max(TA_MVC, sum(Test_data_TA[i:i + Slide_length]))
        TA_MVC = TA_MVC /Slide_length
        # return {"TA_MVC":TA_MVC}
        return {"Avanti sensor 4: EMG 4":TA_MVC}
    elif("GAS" in File_name and "filter"  in File_name):
        Test_data_GAS=np.array(data["Avanti sensor 5: EMG 5"])
        Test_data_SOL=np.array(data["Avanti sensor 6: EMG 6"])
        GAS_MVC=0
        SOL_MVC=0
        for i in range(len(Test_data_GAS) - Slide_length):
            GAS_MVC = max(GAS_MVC, sum(Test_data_GAS[i:i + Slide_length]))
        GAS_MVC = GAS_MVC / Slide_length
        for i in range(len(Test_data_SOL) - Slide_length):
            SOL_MVC = max(SOL_MVC, sum(Test_data_SOL[i:i + Slide_length]))
        SOL_MVC = SOL_MVC / Slide_length
        # return {"GAS_MVC":GAS_MVC,"SOL_MVC":SOL_MVC}
        return {"Avanti sensor 5: EMG 5": GAS_MVC, "Avanti sensor 6: EMG 6": SOL_MVC}
    return 0


def FindMVC_in_File_Speed(File_name):
    data=pd.read_csv(File_name)
    Slide_length = 150
    if(True):
        Test_data_RF=np.array(data["Avanti sensor 1: EMG 1"])
        Test_data_VL=np.array(data["Avanti sensor 2: EMG 2"])
        Test_data_BF=np.array(data["Avanti sensor 3: EMG 3"])
        Test_data_TA=np.array(data["Avanti sensor 4: EMG 4"])
        Test_data_GAS=np.array(data["Avanti sensor 5: EMG 5"])
        Test_data_SOL=np.array(data["Avanti sensor 6: EMG 6"])
        RF_Max=0
        VL_Max=0
        BF_Max = 0
        TA_Max = 0
        GAS_Max=0
        SOL_Max=0
        RF_Min=0
        VL_Min=0
        BF_Min = 0
        TA_Min = 0
        GAS_Min=0
        SOL_Min=0
        for i in range(len(Test_data_RF)-Slide_length):
            RF_Max=max(RF_Max,sum(Test_data_RF[i:i+Slide_length]))
        RF_Max=RF_Max/Slide_length
        for i in range(len(Test_data_VL)-Slide_length):
            VL_Max=max(VL_Max,sum(Test_data_VL[i:i+Slide_length]))
        VL_Max=VL_Max/Slide_length
        # return {"RF_MVC":RF_MVC,"VL_MVC":VL_MVC}
        for i in range(len(Test_data_BF) - Slide_length):
            BF_Max = max(BF_Max, sum(Test_data_BF[i:i + Slide_length]))
        BF_Max = BF_Max / Slide_length
        for i in range(len(Test_data_TA) - Slide_length):
            TA_Max = max(TA_Max, sum(Test_data_TA[i:i + Slide_length]))
        TA_Max = TA_Max /Slide_length
        for i in range(len(Test_data_GAS) - Slide_length):
            GAS_Max = max(GAS_Max, sum(Test_data_GAS[i:i + Slide_length]))
        GAS_Max = GAS_Max / Slide_length
        for i in range(len(Test_data_SOL) - Slide_length):
            SOL_Max = max(SOL_Max, sum(Test_data_SOL[i:i + Slide_length]))
        SOL_Max = SOL_Max / Slide_length

        Slide_length=10
        for i in range(len(Test_data_RF)-Slide_length):
            RF_Min=min(RF_Min,sum(Test_data_RF[i:i+Slide_length]))
        RF_Min=RF_Min/Slide_length
        for i in range(len(Test_data_VL)-Slide_length):
            VL_Min=min(VL_Min,sum(Test_data_VL[i:i+Slide_length]))
        VL_Min=VL_Min/Slide_length
        # return {"RF_MVC":RF_MVC,"VL_MVC":VL_MVC}
        for i in range(len(Test_data_BF) - Slide_length):
            BF_Min = min(BF_Min, sum(Test_data_BF[i:i + Slide_length]))
        BF_Min = BF_Min / Slide_length
        for i in range(len(Test_data_TA) - Slide_length):
            TA_Min = min(TA_Min, sum(Test_data_TA[i:i + Slide_length]))
        TA_Min = TA_Min /Slide_length
        for i in range(len(Test_data_GAS) - Slide_length):
            GAS_Min = min(GAS_Min, sum(Test_data_GAS[i:i + Slide_length]))
        GAS_Min = GAS_Min / Slide_length
        for i in range(len(Test_data_SOL) - Slide_length):
            SOL_Min = min(SOL_Min, sum(Test_data_SOL[i:i + Slide_length]))
        SOL_Min = SOL_Min / Slide_length

        return {"Avanti sensor 1: EMG 1": [RF_Max,RF_Min],"Avanti sensor 2: EMG 2": [VL_Max,VL_Min],
                "Avanti sensor 3: EMG 3":[BF_Max,BF_Min],"Avanti sensor 4: EMG 4":[TA_Max,TA_Min],
                "Avanti sensor 5: EMG 5": [GAS_Max,GAS_Min], "Avanti sensor 6: EMG 6": [SOL_Max,SOL_Min]}
        # return {"EMG1": [RF_Max,RF_Min],"EMG2": [VL_Max,VL_Min],
        #         "EMG3":[BF_Max,BF_Min],"EMG4":[TA_Max,TA_Min],
        #         "EMG5": [GAS_Max,GAS_Min], "EMG6": [SOL_Max,SOL_Min]}


def Calc_Test_MVC(Folder_path):
    File_list=os.listdir(Folder_path)
    MVC_dict={}
    for i in range(len(File_list)):
        # Temp_dict=(FindMVC_in_File(Folder_path+"//"+File_list[i]))

        Temp_dict=(FindMVC_in_File(Folder_path+"//"+File_list[i]))

        # Temp_dict = (FindMVC_in_File_Speed(Folder_path + "//" + File_list[i]))

        if(type(Temp_dict)==dict):
            for i,j in Temp_dict.items():
                if i not in MVC_dict.keys():
                    MVC_dict[i]=[]
                    MVC_dict[i].append(j)
                else:
                    MVC_dict[i].append(j)
    for i,j in MVC_dict.items():
        MVC_dict[i]=sum(j)/len(j)
    return(MVC_dict)

def CompareMVC(File1,File2,MVC_Dict1,MVC_Dict2):
    ####找出步态里面peak  求均值   和MVC对比，作图。

    Ave_EMG1={}
    Ave_EMG2={}
    Ave_EMG1=Calc_EMG_Peakvalue(File1)
    Ave_EMG2=Calc_EMG_Peakvalue(File2)
    MVC1=Calc_Test_MVC(MVC_Dict1)
    MVC2=Calc_Test_MVC(MVC_Dict2)

    MVC1_Percent=[]
    MVC2_Percent = []
    for i,j in Ave_EMG1.items():
           MVC1_Percent.append(j/MVC1[i])
    for i,j in Ave_EMG2.items():
           MVC2_Percent.append(j/MVC2[i])
    return 0





def Dataprocess(File_name,minmax,Dataloader_name):
    Preprocess_Delsys(File_name)
    Interpolate_IMU(File_name+"_filter.csv")
    stride_list = Find_heelstrike_in_IMU(File_name+"_filter.csv_InterpolateIMU.csv")
    # Catch_EMG_Exception_Out(File_name+"_filter.csv_InterpolateIMU.csv", stride_list)
    return 0
    # Minmaxfile(File_name+"_filter.csv_interpolate.csv_EMGexpout.csv",minmax)
    # Load_Data_toDataloader(File_name+"_filter.csv_interpolate.csv_EMGexpout.csv_Minmax.csv",Dataloader_name)

    # addr = r"C:\Users\liyunze\Desktop\20221205\Train_Dataset\1.4ms_3min\EMG_filter.csv_interpolate.csv_percentage.csv_EMGexpout.csv"
    # minmax = (Findminmax_Index(addr))

def Calc_EMG_Peakvalue(File_name):
    data1 = pd.read_csv(File_name)
    strike_num1 = Find_heelstrike_in_IMU(File_name)
    Ave_EMG1 = {}
    EMG_Col_list = ["Avanti sensor 1: EMG 1", "Avanti sensor 2: EMG 2"
        , "Avanti sensor 3: EMG 3", "Avanti sensor 4: EMG 4"
        , "Avanti sensor 5: EMG 5", "Avanti sensor 6: EMG 6"]
    for i in range(len(EMG_Col_list)):
        sum1 = 0
        for j in range(len(strike_num1)):
            sum1 += np.array(data1[EMG_Col_list[i]])[strike_num1[j][0]:strike_num1[j][1]].max()
        Ave_EMG1[EMG_Col_list[i]] = sum1 / len(strike_num1)
    return Ave_EMG1

def Filter_Folder(Folder_name):
    Folder_list=os.listdir(Folder_name)
    for i in range(len(Folder_list)):
        if("filter" not in Folder_list[i] and "csv" in Folder_list[i]):
            Preprocess_Delsys(Folder_name+"\\"+Folder_list[i],Folder_name,Folder_list[i])


def Interpolate_Folder(Folder_name):
    Folder_list=os.listdir(Folder_name)
    for i in range(len(Folder_list)):
        if("csv" in Folder_list[i]):
            Interpolate_IMU(Folder_name+"\\Filtered\\"+Folder_list[i],Folder_name,Folder_list[i])


def Add_Percentage_Folder(Folder_name):
    Folder_list=os.listdir(Folder_name)
    for i in range(len(Folder_list)):
        if("csv" in Folder_list[i]):
            Find_heelstrike_in_IMU(Folder_name,Folder_name+"\\Interpolate\\"+Folder_list[i],Folder_list[i],1)


def Catch_Expout_Folder(Folder_name):
    Folder_list = os.listdir(Folder_name)
    for i in range(len(Folder_list)):
        if("csv" in Folder_list[i]):
            stride_list = Find_heelstrike_in_IMU("None",Folder_name+"\\Interpolate\\"+Folder_list[i])
            Catch_EMG_Exception_Out(Folder_name+"\\Interpolate\\"+Folder_list[i],Folder_name,Folder_list[i],stride_list)

def MinmaxFolder(Folder_Name,IMU_minmax,MVC_file):
    Folder_list = os.listdir(Folder_Name+"\\Percentage")
    Lower_dict=EMGLower_find(MVC_file)
    MVC=FindMVC_in_File_Speed(MVC_file)
    for key,value in Lower_dict.items():
        MVC[key][1]=value
    for j in range(len(Folder_list)):
        data=pd.read_csv(Folder_Name+"\\Percentage"+"\\"+Folder_list[j])
    ###IMU input normalization
        columns=data.columns
        for i in range(len(columns)):
            if("ACC" in str(columns[i]) or "GYRO" in str(columns[i])):
                        mx, mi = IMU_minmax[str(columns[i])]["max"], IMU_minmax[str(columns[i])]["min"]
                        data[columns[i]] = (data[columns[i]] - mi) / (mx - mi)
                        while (1):
                            if (data[columns[i]].max() > 1):
                                data[columns[i]][data[columns[i]].idxmax()] = 1
                            else:
                                break
                        while (1):
                            if (data[columns[i]].min() < 0):
                                data[columns[i]][data[columns[i]].idxmin()] = 0
                            else:
                                break
    ###EMG output normalization
            EMG_Sensors=["Avanti sensor 1: EMG 1","Avanti sensor 2: EMG 2","Avanti sensor 3: EMG 3"
                         ,"Avanti sensor 4: EMG 4","Avanti sensor 5: EMG 5","Avanti sensor 6: EMG 6"]
            if (str(columns[i]) in EMG_Sensors):
                mx, mi = MVC[str(columns[i])][0], MVC[str(columns[i])][1]
                data[columns[i]] = (data[columns[i]] - mi) / (mx - mi)
                while (1):
                    if (data[columns[i]].max() > 1):
                        data[columns[i]][data[columns[i]].idxmax()] = 1
                    else:
                        data[columns[i]] = round(data[columns[i]] * 100)
                        break
                while (1):
                    if (data[columns[i]].min() < 0):
                        data[columns[i]][data[columns[i]].idxmin()] = 0
                    else:

                        break

    ####store
        if(not os.path.exists(Folder_Name+"/Minmax")):
            os.mkdir(Folder_Name+"\\Minmax")
        data.to_csv(Folder_Name+"\\Minmax "+"\\"+Folder_list[j])




def plot1(Folder_path):
    File_list=os.listdir(Folder_path)
    Key_words=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
    EMG_Dict={}
    for i in range(len(File_list)):
        if("1.5" in File_list[i] and "filter" in File_list[i] and "Interpolate" not in File_list[i]):
                Interpolate_IMU(Folder_path+"//"+File_list[i])
                stride_list = Find_heelstrike_in_IMU(Folder_path+"//"+File_list[i]+"_InterpolateIMU.csv")
                Catch_EMG_Exception_Out(Folder_path+"//"+File_list[i]+"_InterpolateIMU.csv", stride_list)
                MVC1=FindMVC_in_File_Speed(Folder_path+"//"+File_list[i]+"_InterpolateIMU.csv"+"_EMGexpout.csv")
        #
        # for j in range(len(Key_words)):
        #     if(str(Key_words[j]) in File_list[i] and "filter" in File_list[i]):
        #         Ave_EMG = Calc_EMG_Peakvalue(Folder_path+"//"+File_list[i])
        #         EMG_Dict[Key_words[j]]=Ave_EMG
    count=1
    # plt_xlabel = []
    # plt_list = []
    # for i,j in MVC1.items():
    #     for k,m in EMG_Dict.items():
    #         plt_xlabel.append(k)
    #         plt_list.append(m[i])
        # plt.subplot(2, 3, count)
        # plt.scatter(plt_xlabel,plt_list)
        # plt.title(i)
        # plt.xticks(plt_xlabel)
        # plt.axhline(j)
        # count+=1
        # plt.legend()
    # plt.show()
    return MVC1

def plot2(Folder_path):
    File_list=os.listdir(Folder_path)
    Key_words=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
    EMG_Dict={}
    MVC1=Calc_Test_MVC(Folder_path)
    for i in range(len(File_list)):
        for j in range(len(Key_words)):
            if(str(Key_words[j]) in File_list[i] and "filter" in File_list[i]):
                Ave_EMG = Calc_EMG_Peakvalue(Folder_path+"//"+File_list[i])
                EMG_Dict[Key_words[j]]=Ave_EMG
    count=1
    for i,j in MVC1.items():

        plt_xlabel=[]
        plt_list=[]
        for k,m in EMG_Dict.items():
            plt_xlabel.append(k)
            plt_list.append(m[i])
        plt.subplot(2, 3, count)
        plt.scatter(plt_xlabel,plt_list)
        plt.title(i)
        plt.xticks(plt_xlabel)
        plt.axhline(j)
        count+=1
        plt.legend()
    plt.show()
    return 0




def FindIMU_Range(IMU_list):
    IMU_Minmax={}
    for i in range(len(IMU_list)):
        temp_data=pd.read_csv(IMU_list[i])
        for j in range(len(temp_data.columns)):
                if("ACC" in temp_data.columns[j] or "GYRO" in temp_data.columns[j]):
                     if(len(IMU_Minmax)<=60):
                         IMU_Minmax[temp_data.columns[j]]={"max":temp_data[temp_data.columns[j]].max(),"min":temp_data[temp_data.columns[j]].min()}

                     # elif(IMU_Minmax[temp_data.columns[j]] not in list(IMU_Minmax.keys())):
                     #     IMU_Minmax[temp_data.columns[j]] = {"max": temp_data[temp_data.columns[j]].max(),"min": temp_data[temp_data.columns[j]].min()}
                     else:
                         IMU_Minmax[temp_data.columns[j]]={"max":max(float(temp_data[temp_data.columns[j]]),IMU_Minmax[temp_data.columns[j]["max"]]),"min":min(float(temp_data[temp_data.columns[j]]),IMU_Minmax[temp_data.columns[j]["min"]])}



    return IMU_Minmax


def Data_processing(Folder_Name):
    Filter_Folder(Folder_Name)
    Interpolate_Folder(Folder_Name)
    Add_Percentage_Folder(Folder_Name)
    Catch_Expout_Folder(Folder_Name)
    return 0


def EMGLower_find(File):
    data=pd.read_csv(File)
    Emg_list=["Avanti sensor 1: EMG 1","Avanti sensor 2: EMG 2","Avanti sensor 3: EMG 3",
              "Avanti sensor 4: EMG 4","Avanti sensor 5: EMG 5","Avanti sensor 6: EMG 6"]
    Low_dict={}
    for i in range(len(Emg_list)):
        raw_data=data[Emg_list[i]]
        filter_data=moving_average(data[Emg_list[i]].tolist(),200).tolist()
        lower_points=find_lower(filter_data)
        sum=0
        for j in range(len(lower_points)):
            sum+=raw_data[lower_points[j]]
        average=sum/len(lower_points)
        Low_dict[Emg_list[i]]=average
    return Low_dict


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def ViewBug(file_list):
    plt_list={}
    Keyword="Avanti sensor 6: EMG 6"
    for i in range(len(file_list)):
         plt_list[i]=pd.read_csv(file_list[i])[Keyword]
    for key,value in plt_list.items():
        plt.plot(range(len(value)),value,label=key)
    plt.figure()
    plt.legend()
    plt.show()
    print("0")

def find_ranges(sequence):
    ranges = []
    start = 0
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            ranges.append((start+1, i))
            start = i
    ranges.append((start, len(sequence)-1))  # add the last range
    return ranges


def Average_Stride_EMG(file_name):
    data=pd.read_csv(file_name)
    columns=data.columns
    stride_dict={}
    # columns[2]=stride
###通过stride 划分步态，得到上届和下届
    stride_flag=(find_ranges(data[columns[2]]))
    for i in range(len(stride_flag)):
        stride_dict[i]={"EMG1":data['Avanti sensor 1: EMG 1'][stride_flag[i][0]:stride_flag[i][1]],
                        "EMG2": data['Avanti sensor 2: EMG 2'][stride_flag[i][0]:stride_flag[i][1]],
                        "EMG3": data['Avanti sensor 3: EMG 3'][stride_flag[i][0]:stride_flag[i][1]],
                        "EMG4": data['Avanti sensor 4: EMG 4'][stride_flag[i][0]:stride_flag[i][1]],
                        "EMG5": data['Avanti sensor 5: EMG 5'][stride_flag[i][0]:stride_flag[i][1]],
                        "EMG6": data['Avanti sensor 6: EMG 6'][stride_flag[i][0]:stride_flag[i][1]]
                        }

    TARGET_LENGTH = 2000
    interpolated_data = {}

    for stride, emgs in stride_dict.items():
        interpolated_data[stride] = {}
        for emg, values in emgs.items():
            # create interpolation function
            f = interp1d(np.linspace(0, 1, len(values)), values, kind='linear')

            # create new data array with target length
            new_values = f(np.linspace(0, 1, TARGET_LENGTH))

            # store interpolated data
            interpolated_data[stride][emg] = new_values


    averages = {}

    # get the EMG names from the first stride
    emg_names = interpolated_data[list(interpolated_data.keys())[0]].keys()

    for emg in emg_names:
        all_values = []

        # gather all values for this EMG
        for stride in interpolated_data.values():
            all_values.append(stride[emg])

        # compute the average
        averages[emg] = np.mean(all_values, axis=0)

    # emg1=averages["EMG6"].tolist()
    #
    # plt.plot(range(len(emg1)),emg1)
    # plt.show()
    return averages



def calculate_emg_rms(emg_signal):
    squared_emg = np.square(emg_signal)  # 将 EMG 信号平方
    mean_squared_emg = np.mean(squared_emg)  # 计算平方后的信号的平均值
    emg_rms = np.sqrt(mean_squared_emg)  # 对平均值开根号得到 RMS 值
    return emg_rms


def Load_Data_toDataloader(Input_file,Output_Loadername):
    ##############['pjo`d
    ##在两次数据清洗后，选取需要的数据放入dataloader
    data = pd.read_csv(Input_file)
    data3=UseDataLoader_Output_Emg_Without_Period(data,50)
    train_loader1=DataLoader(
        dataset=data3,
        batch_size=512,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )

    save_to_pickle(train_loader1, Output_Loadername+"_SHUFFLE")


def Load_Data_toDataloader_noshuffle(Input_file,Output_Loadername):
    ##############['pjo`d
    ##在两次数据清洗后，选取需要的数据放入dataloader
    data = pd.read_csv(Input_file)
    data3=UseDataLoader_Output_Emg_Without_Period(data,50)
    train_loader1=DataLoader(
        dataset=data3,
        batch_size=512,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )

def Merge_Data_toDataloader(Input_file_list, Output_Loadername):
        ##############['pjo`d
        ##在两次数据清洗后，选取需要的数据放入dataloader
        data_raw_1 = pd.read_csv(Input_file_list[0])
        data1 = UseDataLoader_Output_Emg_Without_Period(data_raw_1, 50)
        data_raw_2 = pd.read_csv(Input_file_list[1])
        data2 = UseDataLoader_Output_Emg_Without_Period(data_raw_2, 50)
        combined_dataset = ConcatDataset([data1,data2])

        for i in range(len(Input_file_list)-2):
            data = pd.read_csv(Input_file_list[i+2])
            data3 = UseDataLoader_Output_Emg_Without_Period(data, 50)
            combined_dataset = ConcatDataset([combined_dataset, data3])
            print(len(combined_dataset))

        combined_dataloader=DataLoader(
            dataset=combined_dataset,
            batch_size=512,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0

        )

        save_to_pickle(combined_dataloader, Output_Loadername)



def get_file_paths(folder_name, subfolder_name):
    subfolder_path = os.path.join(folder_name, subfolder_name)
    file_names = os.listdir(subfolder_path)
    file_paths = [os.path.join(subfolder_path, file_name) for file_name in file_names]
    return file_paths


def Data_loader_processing(Folder_Name,User_Name):
    file_paths = get_file_paths(Folder_Name, 'Minmax')
    Merge_Data_toDataloader(file_paths, User_Name)


def Single_Data_loader_processing(Folder_Name):
    file_paths = get_file_paths(Folder_Name, 'Minmax')
    for i in file_paths:
        Load_Data_toDataloader_noshuffle(i,os.path.basename(Folder_Name)+"_"+os.path.basename(i))


if __name__ == '__main__':
    #data processing
    Data_processing(r"0119LYZ")
    #input output normalization
    IMU_Range_list=[ r"0119LYZ\EMGexpout\1.5.csv"]
    IMU_Range=FindIMU_Range(IMU_Range_list)
    MinmaxFolder(r"0119LYZ",IMU_Range,r"0119LYZ\EMGexpout\1.5.csv")

    #load to dataloader
    # creat train dataloader
    Data_loader_processing("0119LYZ","yz")

    # creat validate dataloader
    Single_Data_loader_processing("0119LYZ")