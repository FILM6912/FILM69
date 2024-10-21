import requests
import pandas as pd
import json
from datetime import datetime, timedelta

class ThingsBoard:

  def __init__(self,host:str="thingsboard.weaverbase.com",user:str=None,password=None) -> None:
    self.host=host
    url = f"https://{host}/api/auth/login"
    payload = "{"+f"\"username\": \"{user}\",\"password\": \"{password}\""+"}"
    headers = {
      'Content-Type': 'text/plain'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
      self.token=json.loads(response.content.decode())["token"]
    else:raise ValueError("response code",response.status_code)
    
  def send_data(self,telemetry = {"PM": 0},device_ID=""):
      
      url = f"https://{self.host}:443/api/plugins/telemetry/DEVICE/{device_ID}/timeseries/ANY?scope=ANY"

      payload = json.dumps(telemetry)
      headers = {
      'accept': 'application/json',
      'Content-Type': 'application/json',
      'X-Authorization': f'Bearer {self.token}'
      }
      response = requests.request("POST", url, headers=headers, data=payload)

      # print(response.text)

  def get_data(self,para=list[str],limit=24,startTS="",endTS="",device_ID="",time=None):
      start=int(datetime.strptime(startTS, "%Y-%m-%d %H:%M:%S").timestamp()*1000)
      end=int(datetime.strptime(endTS, "%Y-%m-%d %H:%M:%S").timestamp()*1000)
      
      url = f'https://{self.host}:443/api/plugins/telemetry/DEVICE/{device_ID}/values/timeseries'
      params = {'keys': para, 'startTs': start, 'endTs': end, 'limit': f"{limit}"}
      headers = {'accept': 'application/json', 'X-Authorization': f'Bearer {self.token}'}

      response = requests.get(url, params=params, headers=headers)
      
      if response.status_code == 200:
        formatted_data = {}
        for key, values in json.loads(response.content.decode("utf-8")).items():
            formatted_data[key]=[]
            ts=[]
            for item in values:
                formatted_data[key].append(item["value"])
                ts.append(item["ts"])
        formatted_data["ts"]=ts
        
        max_key = min(formatted_data, key=lambda k: len(formatted_data[k]))
        for i in formatted_data.keys():
          formatted_data[i]=formatted_data[i][:len(formatted_data[max_key])]
        
        df=pd.DataFrame(formatted_data)
        df["ts"]=df["ts"].apply(lambda value:datetime.fromtimestamp(value/1000))
        df['ts'] = pd.to_datetime(df['ts'], dayfirst=True)
        df=df.sort_values("ts").reset_index().drop(columns="index")[["ts"]+para]
        if time!=None:
          df.set_index('ts', inplace=True)
          df=df.resample(time).nearest()
          
          return df.reset_index()
        else:return df
      else:return f"Error{response.status_code}"

if __name__ == "__main__":
  tb=ThingsBoard("thingsboard.weaverbase.com","user@gmail.com","password")
  
  end=datetime.now()
  start = (end-timedelta(days=1))
  startTS=start.strftime("%Y-%m-%d %H:%M:%S")
  endTS=end.strftime("%Y-%m-%d %H:%M:%S")
  
  x=tb.get_data(["Humidity","Soil Moisture_1","Soil Moisture_2",
          "Soil Moisture_3","Soil Moisture_4",
          "Soil Teamperature_1","Soil Teamperature_2"
          ],startTS=startTS,endTS=endTS,device_ID="")

  print(x)