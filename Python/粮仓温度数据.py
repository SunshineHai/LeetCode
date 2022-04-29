import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta


# 得到下一个 sheet_name
def get_next_sheet(date_time: str):
    month = int(date_time[:2])
    day = int(date_time[2:])

    time = datetime(2022, month, day)
    d = time + timedelta(weeks=0, days=1, hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0, )
    # 将日期转换成字符串
    date = d.strftime('%Y%m%d')
    return date[4:]


path = r'C:\Users\JackYang\Desktop\数据处理\12仓.xlsx'


def read_sheet(path, begin_sheet_name: str):

    print(begin_sheet_name)
    data = pd.read_excel(path, sheet_name=begin_sheet_name)

    # 1.日期
    my_str = data.iloc[0, 0]
    data_time = my_str[my_str.index('时') + 3: my_str.index('日') + 1]
    data_time = pd.Series(data_time)

    # 提取一个sheet页的数据
    res = pd.DataFrame()
    for i in [0, 4, 8]:
        # 2.时间
        time = data.iloc[1, 2 + i]
        time = pd.Series(time)
        time

        # 3.外温
        out_temp = data.iloc[2, 2 + i]
        out_temp = pd.Series(out_temp)

        # 4.外湿
        out_wet = data.iloc[3, 2 + i]
        out_wet = pd.Series(out_wet)

        # 5.仓湿
        w_wet = data.iloc[4, 2 + i]
        w_wet = pd.Series(w_wet)

        # 6. 仓1 , 2, 3, 4,5
        w1 = pd.Series(data.iloc[5, 3 + i])
        w2 = pd.Series(data.iloc[6, 3 + i])
        w3 = pd.Series(data.iloc[7, 3 + i])
        w4 = pd.Series(data.iloc[8, 3 + i])
        w5 = pd.Series(data.iloc[9, 3 + i])

        # 7.粮温高 下、中、上、表
        h1 = pd.Series(data.iloc[11, 2 + i])
        h2 = pd.Series(data.iloc[11, 3 + i])
        h3 = pd.Series(data.iloc[11, 4 + i])
        h4 = pd.Series(data.iloc[11, 5 + i])

        # 8 低
        L1 = pd.Series(data.iloc[12, 2 + i])
        L2 = pd.Series(data.iloc[12, 3 + i])
        L3 = pd.Series(data.iloc[12, 4 + i])
        L4 = pd.Series(data.iloc[12, 5 + i])

        # 9 平均
        a1 = pd.Series(data.iloc[13, 2 + i])
        a2 = pd.Series(data.iloc[13, 3 + i])
        a3 = pd.Series(data.iloc[13, 4 + i])
        a4 = pd.Series(data.iloc[13, 5 + i])

        # 10.
        # 连接
        mid = pd.concat([data_time, time, out_temp, out_wet, w_wet
                            , w1, w2, w3, w4, w5,
                         h1, h2, h3, h4,
                         L1, L2, L3, L4,
                         a1, a2, a3, a4], axis=1)
        res = pd.concat([res, mid], axis=0)
        pass
    return res


if __name__ == '__main__':
    save_path = r'C:\Users\JackYang\Desktop\数据处理\12仓_res.xlsx'
    # res = read_sheet(path, "0829")
    # res.to_excel(save_path, index=None)

    final_res = pd.DataFrame()
    begin_date = '0629'                         # 0629
    while begin_date <= "0903":
        print(begin_date)
        res = read_sheet(path, begin_date)
        final_res = pd.concat([final_res, res], axis=0)
        begin_date = get_next_sheet(begin_date)
        pass
    final_res.columns = ["日期", '时间', '外温', '外湿', '仓湿', '仓1', '仓2', '仓3', '仓4', '仓5',
                            '下高', '中高', '上高', '表高', '下低', '中低', '上底', '表低',
                            '下平', '中平', '上平', '表平']
    final_res.to_excel(save_path, index=None)
    # print(final_res)