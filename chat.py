import requests
import json
import re

def get_answer(text, user_key):
    data_send = {
        'query': text,
        'sessionId': user_key,
        'lang': 'ko',
    }

    data_header = {
        'Authorization': 'Bearer dffd67b00f1d4386ad15c8e1f3322d33',
        'Content-Type': 'application/json; charset=utf-8'
    }

    dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'
    try:
        res = requests.post(dialogflow_url, data=json.dumps(data_send), headers=data_header)

        if res.status_code != requests.codes.ok:
            return '오류가 발생했습니다.'

        data_receive = res.json()
        answer = data_receive['result']['fulfillment']['speech']

        if answer == "안녕하세요! 무엇을 도와드릴까요?" or answer == "안녕하세요! 약 드실 시간이에요. 약 드셨나요?":
            answer = user_key + "님, " + answer
            return answer

        date = re.compile('.*\d\d\d\d-\d\d-\d\d.*')
        time = re.compile('.*\d\d:\d\d:\d\d.*')

        m = date.match(answer)
        if m!=None:
            answer = re.sub('\d\d\d\d-\d\d-\d\d\w','', answer)

        m = time.match(answer)
        if m!=None:
            answer = re.sub(r'(\d{2}):(\d{2}):(\d{2})',r'\1시 \2분',answer)


    except Exception as e:
        return '오류가 발생했습니다.'

    return answer
