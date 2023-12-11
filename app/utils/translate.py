import requests
import os


class Translate:
    """
    百度翻译模块
    """

    def __init__(self, api_key, secret_key):
        self._host = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
        self.token = ""
        self.refresh_token()

    def refresh_token(self):
        response = requests.get(self._host)
        if response.ok:
            self.token = response.json()["access_token"]

    def translate(self, text, from_lang="auto", to_lang="en"):
        # 空字符串无需翻译 节省流量
        if text == "":
            return ""
        self.refresh_token()
        try:
            result = self._translate(text, from_lang, to_lang)
            dst = result["result"]["trans_result"][0]["dst"]
            return dst
        except Exception as e:
            return text

    def _translate(self, text, from_lang, to_lang):
        url = (
            "https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token="
            + self.token
        )
        # Build request
        headers = {"Content-Type": "application/json"}
        payload = {"q": text, "from": from_lang, "to": to_lang}
        # Send request
        r = requests.post(url, params=payload, headers=headers)
        result = r.json()
        return result


trans = Translate(
    api_key=os.environ.get("BAIDU_API_KEY", ""),
    secret_key=os.environ.get("BAIDU_SECRET_KEY", ""),
)
