from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pandas as pd
import pickle
from helper import *
from helper2 import *

xgb = pickle.load(open("model.pkl", "rb"))
xgb2 = pickle.load(open("modelx.pkl", "rb"))
hostName = "0.0.0.0"
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>Titatic</title></head>", "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>Machine Learning Model (17012023-01) is up.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        content_length = int(self.headers['Content-Length']) 
        post_data = self.rfile.read(content_length)
        data=json.loads(post_data)
        df = pd.DataFrame(data, index=[0])
        if self.path == '/orange':
            df = preprocessing(df)
            try:
                y_pred = xgb2.predict(df)
            except Exception as e:
                print("Error in prediction" + str(e))
                y_pred = [0]
            
            self.send_header("Content-type", "text/html")
            self.end_headers()
            if y_pred[0] == 1:
                self.wfile.write(bytes("churned", "utf-8"))
            else:
                self.wfile.write(bytes("not churned", "utf-8"))
        else :
            try:
                y_pred = xgb.predict(df)
            except Exception as e:
                print("Error in prediction" + str(e))
                y_pred = [0]
            
            self.send_header("Content-type", "text/html")
            self.end_headers()
            if y_pred[0] == 1:
                self.wfile.write(bytes("Survived", "utf-8"))
            else:
                self.wfile.write(bytes("Dead", "utf-8"))
if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    webServer.server_close()
    print("Server stopped.")