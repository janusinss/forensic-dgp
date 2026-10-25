"""Zero-neural loopback JSON transport probe; bounded synthetic byte responses."""
import hashlib
import http.client
import json
from pathlib import Path
import socket
import threading
import time
import urllib.request

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_spatial_dev_review_v1/transport_diagnostic_post_v1.json'


def main():
    assert not OUT.exists()
    app=FastAPI()
    payload=json.dumps({'payload':'x'*2_000_000},separators=(',',':')).encode()
    @app.post('/probe')
    async def probe(file:UploadFile=File(...)):
        content=await file.read()
        assert content==b'z'*92958
        import asyncio
        await asyncio.sleep(16)
        return JSONResponse({'payload':'x'*2_000_000})
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning'))
    thread=threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True);thread.start()
    start=time.monotonic()
    while not server.started:
        assert thread.is_alive() and time.monotonic()-start<10
        time.sleep(.05)
    result={'complete':False,'payload_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),
            'probe_server_port':port,'neural_gradient_optimizer_VM_calls':0,'cases':[]}
    try:
        boundary='TransportProbe'
        upload=(('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="input.png"\r\nContent-Type: image/png\r\n\r\n').encode()+b'z'*92958+('\r\n--'+boundary+'--\r\n').encode())
        headers={'Content-Type':'multipart/form-data; boundary='+boundary}
        for mode in ('urllib_connection_close','http_client_persistent'):
            row={'mode':mode};t=time.monotonic();body=bytearray()
            try:
                if mode.startswith('urllib'):
                    response=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:%d/probe'%port,data=upload,headers=headers),timeout=40)
                    connection=None
                else:
                    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=40)
                    connection.request('POST','/probe',body=upload,headers={**headers,'Connection':'keep-alive'})
                    response=connection.getresponse()
                row['headers']=dict(response.headers)
                length=int(response.getheader('content-length'))
                try:
                    while len(body)<length:
                        chunk=response.read1(min(65536,length-len(body)))
                        if not chunk:break
                        body.extend(chunk)
                    row['exact_body']=bytes(body)==payload
                finally:
                    response.close()
                    if connection:connection.close()
            except Exception as error:
                row['error']={'type':type(error).__name__,'message':str(error)}
            row.update(bytes_received=len(body),seconds=time.monotonic()-t)
            result['cases'].append(row)
        result['complete']=True
    finally:
        server.should_exit=True;thread.join(timeout=5)
        result['probe_server_terminated']=not thread.is_alive()
        result['seconds']=time.monotonic()-start
        OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
