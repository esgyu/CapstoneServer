import pymysql
import pandas as pd
import numpy as np

def connect_sql():
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )

    cursor = drug_db.cursor(pymysql.cursors.DictCursor)

    files = pd.read_csv('drug_information.csv')

    attrs = [4, 5, 10, 11, 12, 13]
    attr = ['제품코드', '업체명', 'small_이미지', 'pack_img', '용법,용량', '효능,효과']

    cnt=0
    for i in range(files.shape[0]):
        sql = """insert into drug_info(code, drug_name, small_image, pack_image)
                values(%s,%s,%s,%s)
        """
        code = int(files.iloc[i][4])
        drug_name = str(files.iloc[i][5])
        small_image = str(files.iloc[i][10]) if not pd.isna(files.iloc[i][10]) else 'null'
        pack_image = str(files.iloc[i][11]) if not pd.isna(files.iloc[i][11]) else 'null'
        cursor.execute(sql,(code, drug_name, small_image, pack_image))
    drug_db.commit()

def selectQuery(codename) :
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )
    cursor = drug_db.cursor(pymysql.cursors.DictCursor)
    sql = """select code, drug_name, small_image, pack_image from drug_info
                        where code = %s
                """
    cursor.execute(sql, (codename))
    rows = cursor.fetchall()
    return rows

def count_maxlength():
    files = pd.read_csv('drug_information.csv')
    # 업체명 ? , small_이미지 ? , pack_img ?, 용법,용량 ?, 효능,효과 ?
    length = [0, 0, 0, 0, 0]
    index = [0, 0, 0, 0, 0]
    attrs = [5, 10, 11, 12, 13]
    attr = ['업체명', 'small_이미지', 'pack_img', '용법,용량', '효능,효과']

    for i in range(files.shape[0]):
        for j in range(5):
            # print((files.iloc[i][attrs[j]]))
            if (pd.isna(files.iloc[i][attrs[j]])):
                continue
            if length[j] < len(files.iloc[i][attrs[j]]):
                length[j] = max(length[j], len(files.iloc[i][attrs[j]]))
                index[j] = i
    print(length)
    for i in range(5):
        print(files.iloc[index[i]][attrs[i]])

if __name__ == "__main__":
    connect_sql();