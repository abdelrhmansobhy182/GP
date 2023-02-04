



def getUnique(data , data_frame):
    unique = data[data_frame].unique()
    list = []
    counter = 0
    for i in unique:
        if type(i) is str:
            list.append(i)
            list.append(counter)
            counter += 1
    return Convert(list)

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct


def encode (data):
    for i in data:
        dic = getUnique(data , i)
        data.replace({i: dic}, inplace=True)

    return data
