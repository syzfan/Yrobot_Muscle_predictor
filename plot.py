from model import *

def plot_plots(plots_real,plots_predict, titles=None, figsize=(8, 8), fontsize=12, border=0.05):

    num_plots = len(plots_real)
    grid_size = math.ceil(math.sqrt(num_plots))
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=figsize)
    fig.subplots_adjust(wspace=border, hspace=border)
    axes = axes.flat
    for i in range(num_plots):
        axes[i].plot(plots_real[i])
        axes[i].plot(plots_predict[i])
        if titles is not None:
            axes[i].set_title(titles[i], fontsize=fontsize, fontweight='bold')
        axes[i].spines['top'].set_visible(True)
        axes[i].spines['right'].set_visible(True)
        axes[i].spines['bottom'].set_visible(True)
        axes[i].spines['left'].set_visible(True)
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.show()
def Plot_one_muscle_to_multi_tasks(dataloader_name,muscle_number,task_pickle_list):
    lstm1 = torch.load(dataloader_name).to(device)
    plot_real=[]
    plot_predict=[]
    for i in range(len(task_pickle_list)):
        print(task_pickle_list[i])
        data_input = load_pickle_data(task_pickle_list[i])
        output1_long = []
        output2_long = []
        output3_long=[]
        for iteration, data in enumerate(data_input):
            inputs = data[0:512, 0:50, 0:36].to(torch.float32).to(device)
            output1 = lstm1(inputs).to(device)
            output1_list = []
            output2_list = []
            for i in range(512):
                    output1_list.append(torch.argmax(output1[i][49]))
            output2 = data[0:512, 0:50, muscle_number+35:muscle_number+36].to(torch.float32)
            for i in range(512):
                    output2_list.append(int(output2[i][49][0].tolist()))
            for i in range(len(output1_list)):
                output1_long.append((output1_list[i]).tolist())
                output2_long.append(output2_list[i] + 100)
                output3_long.append(output2_list[i])
        output3_long=moving_average(output3_long,200)
        output4_long=moving_average(output1_long,200)
        plot_real.append(output3_long)
        plot_predict.append(output4_long)
    # Plot the plots with titles and borders.
    plot_plots(plot_real, plot_predict, figsize=(10, 10), fontsize=14, border=0.1)
def Validate_accuracy(Raw_EMG, Predicted_EMG):
    Length = len(Raw_EMG)
    Error_Constant = 2
    Right = 0
    for i in range(Length):
        if (abs(Raw_EMG[i] - Predicted_EMG[i]) <= Error_Constant):
            Right += 1
    # return ("Accuracy: " + str(Right / Length))
    return Right,Length
def calculate_emg_rms(emg_signal):
    squared_emg = np.square(emg_signal)  # 将 EMG 信号平方
    mean_squared_emg = np.mean(squared_emg)  # 计算平方后的信号的平均值
    emg_rms = np.sqrt(mean_squared_emg)  # 对平均值开根号得到 RMS 值
    return emg_rms
def rgb_to_hex(r, g, b):
    """
    将RGB十进制值转换为16进制值

    :param r: 红色通道的值（0-255）
    :param g: 绿色通道的值（0-255）
    :param b: 蓝色通道的值（0-255）
    :return: RGB值的16进制表示
    """
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)
def find_onset_offset_double_threshold(emg_data_list, fs=1259, high_threshold=0.2, low_threshold=0.1, min_duration=0.05):



    # 将列表转换为NumPy数组
    emg_data = np.array(emg_data_list)

    high_threshold=np.mean(emg_data)+3*np.std(emg_data)
    low_threshold = np.mean(emg_data) + 1 * np.std(emg_data)

    # 设置最短持续时间（以秒为单位）
    min_samples = int(min_duration * fs)

    # 寻找超过高阈值和低阈值的部分
    above_high_threshold = np.where(emg_data > high_threshold, 1, 0)
    above_low_threshold = np.where(emg_data > low_threshold, 1, 0)

    # 寻找onset（从高阈值以下到高阈值以上的转变）
    onset_indices = np.where(np.diff(above_high_threshold) == 1)[0]

    # 寻找offset（从低阈值以上到低阈值以下的转变）
    offset_indices = np.where(np.diff(above_low_threshold) == -1)[0]

    # # 如果offset在onset之前，删除它
    # if offset_indices[0] < onset_indices[0]:
    #     offset_indices = np.delete(offset_indices, 0)
    #
    # # 如果onset在offset之后，删除它
    # if onset_indices[-1] > offset_indices[-1]:
    #     onset_indices = np.delete(onset_indices, -1)
    #
    # # 确保onset和offset的数量相等
    # assert len(onset_indices) == len(offset_indices)
    #
    # # 删除过短的部分
    # i = 0
    # while i < len(onset_indices):
    #     if offset_indices[i] - onset_indices[i] < min_samples:
    #         onset_indices = np.delete(onset_indices, i)
    #         offset_indices = np.delete(offset_indices, i)
    #     else:
    #         i += 1
    #
    # # 返回结果
    return [(onset, onset / fs, offset, offset / fs) for onset, offset in zip(onset_indices, offset_indices)]
def find_onset_offset(emg_data_list, fs=1259, threshold=33, min_duration=0.05):
    # 将列表转换为NumPy数组
    emg_data = np.array(emg_data_list)

    threshold= np.max(emg_data)*0.25

    # 设置最短持续时间（以秒为单位）
    min_samples = int(min_duration * fs)

    # 寻找超过阈值的部分
    above_threshold = np.where(emg_data > threshold, 1, 0)

    # 寻找onset（从阈值以下到阈值以上的转变）
    onset_indices = np.where(np.diff(above_threshold) == 1)[0]

    # 寻找offset（从阈值以上到阈值以下的转变）
    offset_indices = np.where(np.diff(above_threshold) == -1)[0]

    # # 删除过短的部分
    # i = 0
    # while i < len(onset_indices):
    #     if offset_indices[i] - onset_indices[i] < min_samples:
    #         onset_indices = np.delete(onset_indices, i)
    #         offset_indices = np.delete(offset_indices, i)
    #     else:
    #         i += 1

    # 返回结果
    return [(onset/2000, onset / fs, offset/2000, offset / fs) for onset, offset in zip(onset_indices, offset_indices)]
def Plt_Result(data_raw,model_list,muscle_list,interpolate_file,plt_color,plt_ylim,saveroot,batch_size=512):


    if (not os.path.exists(str(data_raw))):
        os.mkdir(str(data_raw))
    for i in range(len(muscle_list)):
        lstmModel=torch.load(model_list[i]).to(device)
        muscle_num=35+muscle_list[i]
        data3=load_pickle_data(data_raw)
        output1_long = []
        output2_long = []
        output3_long = []
        for iteration, data in enumerate(data3):
            inputs = data[0:batch_size, 0:50, 0:36].to(torch.float32).to(device)
            # inputs = torch.cat((data[0:256, 0:50, 0:36], data[0:256, 0:50, 42:43]), dim=2).to(torch.float32).to(device)
            output1 = lstmModel(inputs).to(device)
            output1_list = []
            output2_list = []
            for m in range(batch_size):
                output1_list.append(torch.argmax(output1[m][49]))
            # print(output1_list)
            output2 = data[0:batch_size, 0:50, muscle_num:muscle_num+1].to(torch.float32)
            for m in range(batch_size):
                output2_list.append(int(output2[m][49][0].tolist()))
            # print(output2_list)
            for m in range(len(output1_list)):
                output1_long.append((output1_list[m]).tolist())

                output3_long.append(output2_list[m])

        output3_long = moving_average(output3_long, 200)
        output4_long = moving_average(output1_long, 200)
        averages = Average_Stride_EMG(interpolate_file, output3_long, output4_long)



        plt.plot(range(len(averages["raw_list"])), averages["raw_list"], label="real value", linestyle='-', linewidth=2,
                 color=plt_color[0])
        plt.plot(range(len(averages["predicted_list"])), averages["predicted_list"], label="predicted value",
                 linestyle='--', linewidth=2, color=plt_color[1])



        print("muscle"+str(muscle_list)+" "+interpolate_file.split("/")[6]+" "+interpolate_file.split("/")[8])

        raw_rms=calculate_emg_rms(averages["raw_list"])
        predicted_rms = calculate_emg_rms(averages["predicted_list"])


        print("raw RMS" + " " + str(raw_rms))
        print("predicted RMS"+" "+str(predicted_rms))

        print("accuracy="+" "+str(1-(abs(raw_rms-predicted_rms)/raw_rms)))

        print(find_onset_offset(averages["raw_list"]))
        print(find_onset_offset(averages["predicted_list"]))
        print("                          ")

        # plt.show()


        plt.ylim(-10,plt_ylim)
        # plt.savefig(saveroot)

        # print(find_onset_offset_double_threshold(averages["raw_list"]))

        # print(calculate_emg_rms(output3_long))
        # print(calculate_emg_rms(output4_long))
        # # output3_long = output3_long[22800:] * max_raw_output3 / max_new_output3
        # # output4_long = output4_long[22800:] * max_raw_output4 / max_new_output4
        # plt.plot(range(len(output3_long)), output3_long)
        # plt.plot(range(len(output4_long)), output4_long)
        # # plt.title("Tibialis Anterior")

        # plt.savefig(str(data_raw) + "//" + str(muscle_loader[i]) + str(muscle_list[i]) + ".png")
        plt.clf()
        #

def Easy_Plot(data_raw,model_list,muscle_list,plt_color,batch_size=512):


    if (not os.path.exists(str(data_raw))):
        os.mkdir(str(data_raw))
    for i in range(len(muscle_list)):
        lstmModel=torch.load(model_list[i]).to(device)
        muscle_num=35+muscle_list[i]
        data3=load_pickle_data(data_raw)
        output1_long = []
        output2_long = []
        output3_long = []
        for iteration, data in enumerate(data3):
            inputs = data[0:batch_size, 0:50, 0:36].to(torch.float32).to(device)
            # inputs = torch.cat((data[0:256, 0:50, 0:36], data[0:256, 0:50, 42:43]), dim=2).to(torch.float32).to(device)
            output1 = lstmModel(inputs).to(device)
            output1_list = []
            output2_list = []
            for m in range(batch_size):
                output1_list.append(torch.argmax(output1[m][49]))
            # print(output1_list)
            output2 = data[0:batch_size, 0:50, muscle_num:muscle_num+1].to(torch.float32)
            for m in range(batch_size):
                output2_list.append(int(output2[m][49][0].tolist()))
            # print(output2_list)
            for m in range(len(output1_list)):
                output1_long.append((output1_list[m]).tolist())

                output3_long.append(output2_list[m])

        output3_long = moving_average(output3_long, 200)
        output4_long = moving_average(output1_long, 200)
        # print(output4_long[0:10500])

        #shangpo
        # output4_long=np.concatenate((output4_long[950:10750],output4_long[21100:27300]))
        # output3_long = np.concatenate((output3_long[950:10750], output3_long[21100:27300]))

        #xiapo
        output4_long=np.concatenate((output4_long[:11040],output4_long[23220:]))
        output3_long = np.concatenate((output3_long[:11040], output3_long[23220:]))

        plt.figure(figsize=(20,2))

        plt.plot(range(len(output3_long)), output3_long, label="real value", linestyle='-', linewidth=1,
                 color=plt_color[0])
        plt.plot(range(len(output4_long)), output4_long, label="predicted value",
                 linestyle='-', linewidth=1, color=plt_color[1])
        # plt.savefig(str(data_raw) + "//" + str(muscle_list[i]) + ".png")
        #
        # plt.axis("off")
        plt.show()
        plt.clf()



def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def savevedio(data_raw,model_list,muscle_list,muscle_loader,batch_size=512):
    # usage:
    # validate_data_list = [r"LYZ0306_lyz_shangpo_test4.csv"]
    # muscle_loader = ['20230308_both_muscle2', '20230308_both_muscle3',
    #                  '20230308_both_muscle4', '20230308_both_muscle5', '20230308_both_muscle6']
    # savevedio(validate_data_list[0], [2, 3, 4, 5, 6], muscle_loader, )

    if (not os.path.exists(str(data_raw))):
        os.mkdir(str(data_raw))
    for i in range(len(muscle_list)):
        lstmModel=torch.load(model_list[i]).to(device)
        muscle_num=35+muscle_list[i]
        data3=load_pickle_data(data_raw)
        output1_long = []
        output2_long = []
        output3_long = []
        for iteration, data in enumerate(data3):
            inputs = data[0:batch_size, 0:50, 0:36].to(torch.float32).to(device)
            # inputs = torch.cat((data[0:256, 0:50, 0:36], data[0:256, 0:50, 42:43]), dim=2).to(torch.float32).to(device)
            output1 = lstmModel(inputs).to(device)
            output1_list = []
            output2_list = []
            for m in range(batch_size):
                output1_list.append(torch.argmax(output1[m][49]))
            # print(output1_list)
            output2 = data[0:batch_size, 0:50, muscle_num:muscle_num+1].to(torch.float32)
            for m in range(batch_size):
                output2_list.append(int(output2[m][49][0].tolist()))
            # print(output2_list)
            for m in range(len(output1_list)):
                output1_long.append((output1_list[m]).tolist())
                output2_long.append(output2_list[m] + 100)
                output3_long.append(output2_list[m])

        max_raw_output3 = max(output3_long)
        max_raw_output4 = max(output1_long)
        # output4_long =  output1_long
        output3_long = moving_average(output3_long, 200)
        output4_long = moving_average(output1_long, 200)
        print(calculate_emg_rms(output3_long))
        print(calculate_emg_rms(output4_long))
        # output3_long=output3_long[1500:11000]
        # output4_long = output4_long[1500:11000]
        plt.plot(range(len(output3_long)),output3_long)
        plt.plot(range(len(output4_long)), output4_long)
        plt.show()
        plt.savefig(str(data_raw) + "//" + str(muscle_loader[i]) + str(muscle_list[i]) + ".png")
        plt.clf()


def plot_plots(plots_real,plots_predict, titles=None, figsize=(8, 8), fontsize=12, border=0.05):

    num_plots = len(plots_real)
    grid_size = math.ceil(math.sqrt(num_plots))
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=figsize)
    fig.subplots_adjust(wspace=border, hspace=border)
    axes = axes.flat
    for i in range(num_plots):
        axes[i].plot(plots_real[i])
        axes[i].plot(plots_predict[i])
        if titles is not None:
            axes[i].set_title(titles[i], fontsize=fontsize, fontweight='bold')
        axes[i].spines['top'].set_visible(True)
        axes[i].spines['right'].set_visible(True)
        axes[i].spines['bottom'].set_visible(True)
        axes[i].spines['left'].set_visible(True)
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)
    plt.show()


def Plot_one_muscle_to_multi_tasks(dataloader_name,muscle_number,task_pickle_list):
    lstm1 = torch.load(dataloader_name).to(device)
    plot_real=[]
    plot_predict=[]
    for i in range(len(task_pickle_list)):
        print(task_pickle_list[i])
        data_input = load_pickle_data(task_pickle_list[i])
        output1_long = []
        output2_long = []
        output3_long=[]
        for iteration, data in enumerate(data_input):
            inputs = data[0:512, 0:50, 0:36].to(torch.float32).to(device)
            output1 = lstm1(inputs).to(device)
            output1_list = []
            output2_list = []
            for i in range(512):
                    output1_list.append(torch.argmax(output1[i][49]))
            output2 = data[0:512, 0:50, muscle_number+35:muscle_number+36].to(torch.float32)
            for i in range(512):
                    output2_list.append(int(output2[i][49][0].tolist()))
            for i in range(len(output1_list)):
                output1_long.append((output1_list[i]).tolist())
                output2_long.append(output2_list[i] + 100)
                output3_long.append(output2_list[i])
        output3_long=moving_average(output3_long,200)
        output4_long=moving_average(output1_long,200)
        plot_real.append(output3_long)
        plot_predict.append(output4_long)
    # Plot the plots with titles and borders.
    plot_plots(plot_real, plot_predict, figsize=(10, 10), fontsize=14, border=0.1)
def Validate_accuracy(Raw_EMG, Predicted_EMG):
    Length = len(Raw_EMG)
    Error_Constant = 2
    Right = 0
    for i in range(Length):
        if (abs(Raw_EMG[i] - Predicted_EMG[i]) <= Error_Constant):
            Right += 1
    # return ("Accuracy: " + str(Right / Length))
    return Right,Length

def calculate_emg_rms(emg_signal):
    squared_emg = np.square(emg_signal)  # 将 EMG 信号平方
    mean_squared_emg = np.mean(squared_emg)  # 计算平方后的信号的平均值
    emg_rms = np.sqrt(mean_squared_emg)  # 对平均值开根号得到 RMS 值
    return emg_rms