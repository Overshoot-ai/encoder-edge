import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

from client import StreamingImageClient

HTML = b"""<!doctype html><meta name="viewport" content="width=device-width"><title>Optimized Cross-device Gemma</title>
<style>body{max-width:760px;margin:50px auto;padding:0 20px;font:16px system-ui;color:#171717}input,textarea,button{font:inherit}textarea{box-sizing:border-box;width:100%;min-height:90px;margin:8px 0 14px;padding:10px}button{padding:10px 16px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}.metric{padding:12px;background:#e9f0eb}.metric b{display:block;font-size:20px;margin-top:5px}pre{white-space:pre-wrap}#answer{min-height:80px;padding:16px;background:#102218;color:#ecfff2}#details{color:#555;font-size:13px}@media(max-width:600px){.metrics{grid-template-columns:repeat(2,1fr)}}</style>
<h1>Optimized Cross-device Gemma</h1><p>Local image embeddings, vLLM execution, and SSE token streaming.</p>
<input id="image" type="file" accept="image/*"><textarea id="question">What is shown in this image?</textarea><button id="ask">Ask</button>
<div class="metrics"><div class="metric">Client TTFT<b id="ttft">-</b></div><div class="metric">MPS Encode<b id="encode">-</b></div><div class="metric">Remote TTFT<b id="remote">-</b></div><div class="metric">Tokens<b id="tokens">-</b></div></div><pre id="answer"></pre><pre id="details"></pre>
<script>const byId=id=>document.getElementById(id),ask=byId('ask'),image=byId('image'),question=byId('question'),answer=byId('answer'),ttft=byId('ttft'),encode=byId('encode'),remote=byId('remote'),tokens=byId('tokens'),details=byId('details');ask.onclick=async()=>{if(!image.files[0])return alert('Choose an image');ask.disabled=true;answer.textContent='';let r=new FileReader();r.onload=async()=>{let sent=performance.now(),first=null;try{let x=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:r.result,question:question.value})}),reader=x.body.getReader(),decoder=new TextDecoder(),buffer='';while(true){let q=await reader.read();if(q.done)break;buffer+=decoder.decode(q.value,{stream:true});let lines=buffer.split('\\n');buffer=lines.pop();for(let line of lines){if(!line)continue;let j=JSON.parse(line);if(j.type==='token'){answer.textContent+=j.text;if(first===null){first=performance.now();ttft.textContent=(first-sent).toFixed(1)+' ms'}}else{encode.textContent=j.client_encode_ms.toFixed(1)+' ms';remote.textContent=j.remote_ttft_ms.toFixed(1)+' ms';tokens.textContent=j.usage?.completion_tokens??'-';details.textContent=`pipeline TTFT: ${j.pipeline_ttft_ms.toFixed(1)} ms\npipeline E2E: ${j.pipeline_e2e_ms.toFixed(1)} ms\npreprocess: ${j.client_preprocess_ms.toFixed(1)} ms\nclient encode: ${j.client_encode_ms.toFixed(1)} ms\nserialize: ${j.client_serialize_ms.toFixed(1)} ms\nremote E2E: ${j.remote_e2e_ms.toFixed(1)} ms\nvisual tokens: ${j.visual_tokens}\nrequest: ${(j.request_bytes/1048576).toFixed(2)} MiB`}}}}catch(e){answer.textContent=String(e)}finally{ask.disabled=false}};r.readAsDataURL(image.files[0])}</script>"""


def create_handler(client: StreamingImageClient):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML)))
            self.end_headers()
            self.wfile.write(HTML)

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            image = Image.open(
                io.BytesIO(base64.b64decode(request["image"].split(",", 1)[1]))
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for event in client.stream(image, request["question"]):
                self.wfile.write(json.dumps(event).encode() + b"\n")
                self.wfile.flush()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="gemma-4-12b-optimized")
    parser.add_argument("--port", type=int, default=3001)
    args = parser.parse_args()
    client = StreamingImageClient(args.artifact, args.server, args.model)
    HTTPServer(("127.0.0.1", args.port), create_handler(client)).serve_forever()


if __name__ == "__main__":
    main()
