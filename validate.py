from model import *



data3 = load_pickle_data('1229_1')
output1_long = []
output2_long = []
output3_long=[]
for iteration, data in enumerate(data3):
    inputs = data[0:512, 0:50, 0:36].to(torch.float32).to(device)
    # inputs = torch.cat((data[0:256, 0:50, 0:36], data[0:256, 0:50, 42:43]), dim=2).to(torch.float32).to(device)
    output1 = lstm1(inputs).to(device)
    output1_list = []
    output2_list = []

    for i in range(512):
            output1_list.append(torch.argmax(output1[i][49]))

    # print(output1_list)
    output2 = data[0:512, 0:50, 40:41].to(torch.float32)
    for i in range(512):
            output2_list.append(int(output2[i][49][0].tolist()))
    # print(output2_list)
    for i in range(len(output1_list)):
        output1_long.append((output1_list[i]).tolist())
        output2_long.append(output2_list[i] + 100)
        output3_long.append(output2_list[i])


output3_long=moving_average(output3_long,200)
output4_long=moving_average(output1_long,200)
print(len(output4_long))
plt.plot(range(len(output3_long)),output3_long,label="Real")
plt.plot(range(len(output4_long)),output4_long,label="Predicted")
plt.legend()
plt.show()