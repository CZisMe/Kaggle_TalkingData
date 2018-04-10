import pandas as pd
from sklearn.utils import resample
from sklearn.utils import shuffle 
def balance(address):
    df = pd.read_csv(address)
    df_majority = df[df['is_attributed']==0]
    df_minority = df[df['is_attributed']==1]
    major_val = len(df_majority)
    minor_val = len(df_minority)
    
    df_majority_upsampled = resample(df_majority, 
                                     replace=True,    
                                     n_samples=minor_val,   
                                     random_state=123)
    print(len(df_majority_upsampled), " ", minor_val)
    df_upsampled = pd.concat([df_minority, df_majority_upsampled])
    newdata = shuffle(df_upsampled)
    newdata.to_csv("train_resampled.csv")
def uniqueval():
    data = pd.read_csv("train_sample.csv")
    ipaddress_train = data['ip']
    appid_train = data['app']
    channels_train = data['channel']
    print("training set ip address unique values: ", len(ipaddress_train.unique()))
    print("training set app unique values: ", len(appid_train.unique()))
    print("training set channel unique values: ", len(channels_train.unique()))
    
    data = pd.read_csv("train_sample.csv")
    ipaddress_test = data['ip']
    appid_test = data['app']
    channels_test = data['channel']
    print("training set ip address unique values: ", len(ipaddress_test.unique()))
    print("training set app unique values: ", len(appid_test.unique()))
    print("training set channel unique values: ", len(channels_test.unique()))
    
    print("train_test ipaddress compare: ", ipaddress_test.equals(ipaddress_train))
    print("train_test appid compare: ", appid_test.equals(appid_train))
    print("train_test channels compare: ", channels_test.equals(channels_train))
    
    
    ip_inter = pd.Series(np.intersect1d(ipaddress_train, ipaddress_test))
    app_inter = pd.Series(np.intersect1d(appid_train, appid_test))
    channel_inter = pd.Series(np.intersect1d(channels_train, channels_test))
    print("training set ip intersection: ", len(ip_inter))
    print("training set app intersection: ", len(app_inter))
    print("training set channel intersection: ", len(channel_inter))
