import pymysql
import pandas as pd
import re

def connect_sql():
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )

    cursor = drug_db.cursor(pymysql.cursors.DictCursor)

    files = pd.read_csv('dbdata/drug_info.csv')

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


def update_sql_dose():
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )

    cursor = drug_db.cursor(pymysql.cursors.DictCursor)
    sql = 'select * from drug_info where usages LIKE "%일%회%"'
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        code = row['code']
        usages = row['usages']
        dose_day = re.findall('1일\s+\d+회',usages)
        dose_one = re.findall('1회\s+\d+\w+',usages)
        daily_dose = 1
        single_dose = 1
        if dose_day:
            daily_dose = re.sub(r'1일\s+([0-9]+)회', r'\1', dose_day[0])
        if dose_one:
            single_dose = re.sub(r'1회\s+([0-9]+)\w+',r'\1', dose_one[0])
        sql = 'update drug_info set daily_dose = {}, single_dose ={} where code = {}'.\
            format(daily_dose, single_dose, code)
        cursor.execute(sql)

    drug_db.commit()


def selectQuery(codename):
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )
    cursor = drug_db.cursor(pymysql.cursors.DictCursor)
    sql = """select * from drug_info
                        where code = %s
                """
    cursor.execute(sql, (codename))
    rows = cursor.fetchall()
    return rows

def save_name_code():
    drug_db = pymysql.connect(
        user='root',
        passwd='root',
        host='127.0.0.1',
        db='drug_information',
        charset='utf8'
    )

    cursor = drug_db.cursor(pymysql.cursors.DictCursor)
    sql = 'select * from drug_info'
    cursor.execute(sql)
    rows = cursor.fetchall()
    data = pd.DataFrame(columns=['Code', 'Name'])
    cnt=0
    for row in rows:
        data.loc[cnt] = [row['code'], row['drug_name']]
        cnt+=1
    data.to_csv('save.csv', encoding='utf-8-sig')


if __name__ == "__main__":
    save_name_code()
