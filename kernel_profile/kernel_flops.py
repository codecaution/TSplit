import matplotlib.pyplot as plt


def read_data(dir, layer_type = None):
    input_file = open(dir, 'r')
    rowid = 0
    # if layer_type == "Conv":
    #     # batch_size, 64, 32, 32
    #     # 128, 64, 3, 3
    #     Basic_Flops = 128 * 64 * 3 * 3 * 32 * 32
    # elif layer_type == "Bn":
    #     # Batch_size, 64, 32, 32
    #     Basic_Flops = 
    # Normalized to batch size = 1 as 1.
    batch_size_list = list()
    time_list = list()
    for line in input_file.readlines():
        rowid += 1
        line = line.strip()
        line = line.split(" ")
        if rowid > 4:
            # print(line)
            batch_size_list.append((int)(line[3]))
            time_list.append((float)(line[8]))
            # print(line[3], line[8])

    return batch_size_list, time_list      
    
if __name__ == "__main__":

    batch_size_list, time_list = read_data("./conv_exp/convForward_batch_size_2.txt")
    normalized_time_list = list()
    for idx in range(len(time_list)):
        if idx == 0:
            normalized_time_list.append((float)(1))
        else:
            normalized_time_list.append((float)((time_list[0] * batch_size_list[idx]) / (batch_size_list[0] * time_list[idx])))

    
    plt.title("Normalized Throughput")
    plt.plot(batch_size_list, normalized_time_list, color = 'lightsteelblue')
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput")
    # plt.savefig("./conv_exp/convForward_32x32.jpg")
    plt.savefig("./conv_exp/convForward_batch_size2.jpg")
    # for op_type in ['conv', 'bn', 'relu']:
    #     for t in ['batch_size', 'channel_size', 'height_width']:
    #         input_file = open( op_type + "_" + t + '.txt', 'r')
    #         batch_size_list = []
    #         time_list = []
    #         for line in input_file.readlines():
    #             line = line.strip()
    #             data = line.split(" ")
    #             batch_size_list.append((int)(data[0]))
    #             time_list.append((float)(data[1]))
    #         # print(batch_size_list)
    #         # print(time_list)
    #         if t == 'height_width':
    #             throughput = [(batch_size_list[i] * 1000.0)**2 / time_list[i] for i in range(len(time_list))]
    #         else:
    #             throughput = [batch_size_list[i] * 1000.0 / time_list[i] for i in range(len(time_list))]
    #         # print(throughput)
    #         plt.title("throughput with " + t + "(" + op_type + ')')  
    #         plt.bar(range(len(batch_size_list)), throughput, color = 'lightsteelblue', tick_label = batch_size_list)
    #         plt.savefig('./experiments/' + op_type + '_' + t + '.jpg')
    #         plt.clf()
    
    