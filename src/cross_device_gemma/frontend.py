import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

from .client import ImageClient

HTML = b"""<!doctype html><meta name="viewport" content="width=device-width"><title>Cross-device Gemma</title>
<style>body{max-width:760px;margin:50px auto;padding:0 20px;font:16px system-ui;color:#171717}input,textarea,button{font:inherit}textarea{box-sizing:border-box;width:100%;min-height:90px;margin:8px 0 14px;padding:10px}button{padding:10px 16px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}.metric{padding:12px;background:#eee}.metric b{display:block;font-size:20px;margin-top:5px}pre{white-space:pre-wrap}#answer{min-height:80px;padding:16px;background:#111;color:#eee}#details{color:#555;font-size:13px}@media(max-width:600px){.metrics{grid-template-columns:repeat(2,1fr)}}</style>
<h1>Cross-device Gemma</h1><p>The image is embedded locally. Only visual features and your question are sent to the H200.</p>
<input id="image" type="file" accept="image/*"><textarea id="question">What is shown in this image?</textarea><button id="ask">Ask</button>
<div class="metrics"><div class="metric">Client E2E<b id="client">-</b></div><div class="metric">GPU TTFT<b id="ttft">-</b></div><div class="metric">GPU E2E<b id="gpu">-</b></div><div class="metric">Decode<b id="decode">-</b></div></div><pre id="answer"></pre><pre id="details"></pre>
<script>ask.onclick=async()=>{if(!image.files[0])return alert('Choose an image');ask.disabled=true;answer.textContent='Working...';let r=new FileReader();r.onload=async()=>{try{let x=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:r.result,question:question.value})});let j=await x.json(),m=j.metrics;answer.textContent=j.answer||j.error;if(m){client.textContent=m.client_e2e_ms.toFixed(1)+' ms';ttft.textContent=m.gpu_ttft_ms.toFixed(1)+' ms';gpu.textContent=m.gpu_e2e_ms.toFixed(1)+' ms';decode.textContent=m.decode_tokens_per_second.toFixed(1)+' tok/s';details.textContent=`preprocess: ${m.client_preprocess_ms.toFixed(1)} ms\nclient encode: ${m.client_encode_ms.toFixed(1)} ms\nserialize: ${m.client_serialize_ms.toFixed(1)} ms\nrequest E2E: ${m.request_e2e_ms.toFixed(1)} ms\nserver receive: ${m.server_receive_ms.toFixed(1)} ms\nserver prepare: ${m.server_prepare_ms.toFixed(1)} ms\nvisual tokens: ${m.visual_tokens}\ngenerated tokens: ${m.generated_tokens}\nrequest: ${(m.request_bytes/1048576).toFixed(2)} MiB`}}catch(e){answer.textContent=String(e)}finally{ask.disabled=false}};r.readAsDataURL(image.files[0])}</script>"""


def create_handler(client: ImageClient):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self.respond(200, "text/html; charset=utf-8", HTML)

        def do_POST(self) -> None:
            try:
                started = time.perf_counter()
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                image = Image.open(
                    io.BytesIO(base64.b64decode(request["image"].split(",", 1)[1]))
                )
                result = client.ask(image, request["question"])
                result["metrics"]["browser_bridge_e2e_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                body = json.dumps(
                    {"answer": result["answer"], "metrics": result["metrics"]}
                ).encode()
                self.respond(200, "application/json", body)
            except Exception as error:
                self.respond(
                    400, "application/json", json.dumps({"error": str(error)}).encode()
                )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local browser frontend")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="gemma-4-12b")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    client = ImageClient(args.artifact, args.server, args.model)
    HTTPServer(("127.0.0.1", args.port), create_handler(client)).serve_forever()
