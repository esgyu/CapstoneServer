import pymysql
import pandas as pd


def connect_sql():
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )

    cursor = drug_db.cursor(pymysql.cursors.DictCursor)

    files = pd.read_csv('hello2.csv')

    for i in range(files.shape[0]):
        sql = """insert into drug_info(code, drug_name, small_image, pack_image)
                values(%s,%s,%s,%s)
        """
        code = int(files.iloc[i][4])
        drug_name = str(files.iloc[i][5])
        small_image = str(files.iloc[i][10]) if not pd.isna(files.iloc[i][10]) else 'null'
        pack_image = str(files.iloc[i][11]) if not pd.isna(files.iloc[i][11]) else 'null'
        cursor.execute(sql, (code, drug_name, small_image, pack_image))
    drug_db.commit()


def update_sql():
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )

    cursor = drug_db.cursor(pymysql.cursors.DictCursor)

    files = pd.read_csv('hello2.csv')

    attrs = [4, 5, 10, 11, 12, 13]
    attr = ['제품코드', '업체명', 'small_이미지', 'pack_img', '용법,용량', '효능,효과']

    cnt = 0
    for i in range(files.shape[0]):
        code = int(files.iloc[i][4])
        usages = str(files.iloc[i][12])
        effect = str(files.iloc[i][13])
        # usage = usage.replace('\xa0','')
        # effect = effect.replace('\xa0','')

        sql = 'update drug_info set usages = "{}", effect = "{}" where code = {}'.format(usages, effect, code)
        # print(sql)
        cursor.execute(sql)
    drug_db.commit()


def update_sql_name():
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )

    cursor = drug_db.cursor(pymysql.cursors.DictCursor)

    files = pd.read_csv('hello2.csv')

    for i in range(files.shape[0]):
        code = int(files.iloc[i][4])
        drug_name = str(files.iloc[i][5])

        sql = 'update drug_info set drug_name = "{}" where code = {}'.format(drug_name, code)
        cursor.execute(sql)
    drug_db.commit()


def selectQuery(codename):
    if codename == 'maybe':
        return None

    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )
    cursor = drug_db.cursor(pymysql.cursors.DictCursor)
    sql = """select code, drug_name, small_image, pack_image, usages, effect from drug_info
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
            if (pd.isna(files.iloc[i][attrs[j]])):
                continue
            if length[j] < len(files.iloc[i][attrs[j]]):
                length[j] = max(length[j], len(files.iloc[i][attrs[j]]))
                index[j] = i
    print(length)
    for i in range(5):
        print(files.iloc[index[i]][attrs[i]])


if __name__ == "__main__":
    list = [671803511, 642101970, 643900710, 645403740, 640900250, 644302570, 644306170]
    for i in list:
        print(selectQuery(i))
