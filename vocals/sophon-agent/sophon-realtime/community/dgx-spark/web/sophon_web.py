#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sophon-Realtime 全双工语音网页前端（单文件，普通用户运行）

作用：
  1) 提供一个语音对话网页（麦克风采集、24kHz PCM、server VAD、barge-in 打断）
  2) 把网页的 WebSocket 反向代理到后端 Sophon 服务 ws://127.0.0.1:8000

依赖：aiohttp（vLLM 0.30 已自带，无需额外安装）

本脚本只托管网页 + 反代 WebSocket，不加载模型权重。
权重由另一个进程加载：sophon-realtime serve --model ... --port 8000

远程访问（NVIDIA Sync，当前环境；Sync 已管 SSH/端口转发，不用再跑 tailscale serve）：
  # 1) Spark 上启动本脚本（默认 127.0.0.1:8443）
  source ~/sophon/venv/bin/activate
  python ~/sophon_web.py

  # 2) NVIDIA Sync 里把远端 8443 映射到笔记本本地 8443
  # 3) 笔记本浏览器打开 http://localhost:8443
  #    localhost 属安全上下文，麦克风可用，无需证书

备选 Tailscale Serve（想要 *.ts.net 真实 HTTPS 时）：
  tailscale serve --bg --https=443 http://127.0.0.1:8443
  # 然后打开 https://<机器名>.<tailnet>.ts.net

不用以上两种时可自签 HTTPS：
  python ~/sophon_web.py --self-signed
  # 本机 https://localhost:8443 ；远程 https://<IP>:8443（需信任自签证书）
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import socket
import ssl
import subprocess
import sys
from pathlib import Path

try:
    from aiohttp import web, ClientSession, WSMsgType
except ImportError:
    sys.exit("缺少 aiohttp：先执行 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple aiohttp")


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>智子 · 实时语音</title>
<style>
  :root{
    --bg:#0b0e14; --panel:#141a24; --panel2:#1b2330; --line:#263041;
    --txt:#e6edf7; --dim:#8b98ad; --accent:#4dd4ac; --accent2:#4d9fff;
    --danger:#ff6b6b; --warn:#ffc857;
  }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  html,body{height:100%;margin:0}
  body{background:radial-gradient(1200px 600px at 50% -10%,#16202e 0%,var(--bg) 60%);
    color:var(--txt);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
    display:flex;flex-direction:column;align-items:center;padding:18px 14px 26px}
  .wrap{width:100%;max-width:760px}
  header{display:flex;align-items:center;gap:12px;margin-bottom:14px}
  .logo{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,var(--accent),var(--accent2));
    display:grid;place-items:center;font-weight:800;font-size:19px;color:#06231c}
  h1{font-size:18px;margin:0;font-weight:650;letter-spacing:.3px}
  .sub{font-size:12px;color:var(--dim)}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden}
  .statusbar{display:flex;align-items:center;gap:10px;padding:12px 16px;background:var(--panel2);border-bottom:1px solid var(--line)}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--dim);transition:.25s;flex:none}
  .dot.on{background:var(--accent);box-shadow:0 0 0 4px rgba(77,212,172,.15)}
  .dot.listen{background:var(--accent2);box-shadow:0 0 0 4px rgba(77,159,255,.18)}
  .dot.speak{background:var(--warn);box-shadow:0 0 0 4px rgba(255,200,87,.18)}
  .dot.err{background:var(--danger)}
  #statusText{font-size:13px;color:var(--dim)}
  .meter{margin-left:auto;width:96px;height:7px;background:#0c1119;border-radius:6px;overflow:hidden;flex:none}
  .meter i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .08s}
  #transcript{height:min(48vh,420px);overflow-y:auto;padding:16px}
  .msg{margin:0 0 14px;display:flex;gap:9px;align-items:flex-start}
  .msg .who{font-size:11px;color:var(--dim);flex:none;width:36px;padding-top:3px;text-align:right}
  .msg .bubble{background:var(--panel2);border:1px solid var(--line);padding:8px 12px;border-radius:12px;max-width:82%;white-space:pre-wrap;word-break:break-word}
  .msg.user .bubble{background:rgba(77,159,255,.1);border-color:rgba(77,159,255,.25)}
  .msg.bot .bubble{background:rgba(77,212,172,.09);border-color:rgba(77,212,172,.22)}
  .msg.sys .bubble{background:none;border:none;color:var(--dim);font-size:12px;padding:4px 0}
  .empty{color:var(--dim);text-align:center;padding:38px 10px;font-size:13px}
  .controls{display:flex;gap:10px;padding:14px 16px;border-top:1px solid var(--line);align-items:center;flex-wrap:wrap}
  button{font:inherit;cursor:pointer;border:none;border-radius:11px;padding:11px 20px;font-weight:600;transition:.15s}
  button:disabled{opacity:.4;cursor:not-allowed}
  .primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#06231c;flex:1;min-width:150px}
  .primary:active{transform:scale(.98)}
  .ghost{background:var(--panel2);color:var(--txt);border:1px solid var(--line)}
  select,input{font:inherit;background:var(--panel2);color:var(--txt);border:1px solid var(--line);
    border-radius:9px;padding:8px 10px;outline:none}
  input:focus,select:focus{border-color:var(--accent)}
  details{margin-top:12px;background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:10px 16px}
  summary{cursor:pointer;color:var(--dim);font-size:13px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}
  .grid label{font-size:12px;color:var(--dim);display:flex;flex-direction:column;gap:5px}
  #log{margin-top:10px;max-height:150px;overflow:auto;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
    color:var(--dim);background:#0c1119;border-radius:9px;padding:9px 11px;white-space:pre-wrap;word-break:break-all}
  .tip{font-size:12px;color:var(--dim);margin:12px 2px 0}
  kbd{background:var(--panel2);border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:11px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">智</div>
    <div>
      <h1>智子 · 实时语音</h1>
      <div class="sub">Sophon-Realtime-32B-Tag · 全双工对话</div>
    </div>
  </header>

  <div class="card">
    <div class="statusbar">
      <span class="dot" id="dot"></span>
      <span id="statusText">未连接</span>
      <div class="meter"><i id="meterBar"></i></div>
    </div>
    <div id="transcript"><div class="empty">点「开始通话」，然后直接说话<br>它会边听边想，你随时可以插话打断</div></div>
    <div class="controls">
      <button class="primary" id="btnStart">开始通话</button>
      <button class="ghost" id="btnStop" disabled>结束</button>
      <select id="voice" title="音色">
        <option value="vivian_zh">vivian_zh（女·中）</option>
        <option value="chelsie">chelsie（默认）</option>
        <option value="chelsie_zh">chelsie_zh</option>
        <option value="chelsie_en">chelsie_en</option>
        <option value="brand_zh">brand_zh（男·中）</option>
        <option value="brand_en">brand_en</option>
        <option value="ryan_en">ryan_en（男·英）</option>
      </select>
    </div>
  </div>

  <details>
    <summary>高级设置 &nbsp;·&nbsp; 事件日志</summary>
    <div class="grid">
      <label>模型名<input id="model" value="sophon-realtime"></label>
      <label>输出采样率<select id="rate"><option>24000</option><option>16000</option></select></label>
      <label>VAD 灵敏度<input id="threshold" type="number" min="0" max="1" step="0.05" value="0.5"></label>
      <label>静音判定(ms)<input id="silence" type="number" min="200" max="2000" step="100" value="500"></label>
    </div>
    <div id="log"></div>
  </details>

  <p class="tip">远程访问需 HTTPS（本页已是 <kbd>wss</kbd>）。自签证书首次会提示不安全，点「继续访问」即可。麦克风权限请允许。</p>
</div>

<script>
(function(){
  'use strict';
  const $ = s => document.querySelector(s);
  const dot=$('#dot'), statusText=$('#statusText'), meterBar=$('#meterBar');
  const transcript=$('#transcript'), btnStart=$('#btnStart'), btnStop=$('#btnStop');
  const logBox=$('#log');

  let ws=null, audioCtx=null, micStream=null, procNode=null, sinkGain=null;
  let started=false, listening=false, int16ToFloat=1/32768;
  let activeSources=new Set(), nextPlayTime=0;
  let userTextEl=null, botTextEl=null, lastUserText='';
  const REC_RATE=24000;                 // 输入固定 24kHz（vLLM-Omni realtime 标准）
  let PLAY_RATE=24000;                  // 输出采样率，可在设置里改

  function log(s){
    const t=new Date().toLocaleTimeString('zh-CN',{hour12:false});
    logBox.textContent += `[${t}] ${s}\n`;
    logBox.scrollTop=logBox.scrollHeight;
  }
  function setStatus(kind,text){
    dot.className='dot'+(kind?' '+kind:'');
    statusText.textContent=text;
  }
  function clearEmpty(){ const e=transcript.querySelector('.empty'); if(e) e.remove(); }

  function addMsg(role,text){
    clearEmpty();
    let el;
    if(role==='user'){
      if(!userTextEl || userTextEl.dataset.closed==='1'){
        el=document.createElement('div'); el.className='msg user';
        el.innerHTML='<div class="who">你</div><div class="bubble"></div>';
        transcript.appendChild(el); userTextEl=el.querySelector('.bubble');
      } else { el=userTextEl.parentElement; }
      userTextEl.textContent=(userTextEl.textContent||'')+text;
      lastUserText=userTextEl.textContent;
    } else if(role==='bot'){
      if(!botTextEl || botTextEl.dataset.closed==='1'){
        el=document.createElement('div'); el.className='msg bot';
        el.innerHTML='<div class="who">智子</div><div class="bubble"></div>';
        transcript.appendChild(el); botTextEl=el.querySelector('.bubble');
      }
      botTextEl.textContent=(botTextEl.textContent||'')+text;
    } else {
      el=document.createElement('div'); el.className='msg sys';
      el.innerHTML='<div class="bubble"></div>';
      el.querySelector('.bubble').textContent=text;
      transcript.appendChild(el);
    }
    transcript.scrollTop=transcript.scrollHeight;
  }
  function closeTurns(){
    if(userTextEl) userTextEl.dataset.closed='1';
    if(botTextEl) botTextEl.dataset.closed='1';
    userTextEl=botTextEl=null;
  }

  // ---------- 音频工具 ----------
  function base64ToPcm16(b64){
    const bin=atob(b64), u=new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
    return new Int16Array(u.buffer);
  }
  function pcm16ToBase64(floatArr){
    const n=floatArr.length, out=new Int16Array(n);
    for(let i=0;i<n;i++){
      let s=Math.max(-1,Math.min(1,floatArr[i]));
      out[i]=s<0?s*0x8000:s*0x7fff;
    }
    const bytes=new Uint8Array(out.buffer);
    let str=''; const CH=0x8000;
    for(let i=0;i<bytes.length;i+=CH) str+=String.fromCharCode.apply(null,bytes.subarray(i,i+CH));
    return btoa(str);
  }
  function resample(input,from,to){
    if(from===to) return input;
    const ratio=from/to, out=new Float32Array(Math.round(input.length/ratio));
    for(let i=0;i<out.length;i++){
      const p=i*ratio, i0=Math.floor(p), i1=Math.min(i0+1,input.length-1), f=p-i0;
      out[i]=input[i0]*(1-f)+input[i1]*f;
    }
    return out;
  }
  function playPcm(int16){
    if(!audioCtx) return;
    const buf=audioCtx.createBuffer(1,int16.length,PLAY_RATE);
    const ch=buf.getChannelData(0);
    for(let i=0;i<int16.length;i++) ch[i]=int16[i]*int16ToFloat;
    const src=audioCtx.createBufferSource(); src.buffer=buf; src.connect(sinkGain);
    const now=audioCtx.currentTime;
    const when=nextPlayTime>now?nextPlayTime:now;
    src.start(when); nextPlayTime=when+buf.duration;
    activeSources.add(src);
    src.onended=()=>activeSources.delete(src);
  }
  function stopPlayback(){
    activeSources.forEach(s=>{try{s.stop(0)}catch(e){}});
    activeSources.clear();
    nextPlayTime=audioCtx?audioCtx.currentTime:0;
  }

  // ---------- WebSocket ----------
  function send(obj){ if(ws&&ws.readyState===1) ws.send(JSON.stringify(obj)); }

  function sessionUpdate(){
    const rh=parseInt($('#rate').value,10); PLAY_RATE=rh;
    const session={
      model:$('#model').value||'sophon-realtime',
      audio:{input:{
        format:{type:'audio/pcm',rate:REC_RATE},
        turn_detection:{
          type:'server_vad',
          threshold:parseFloat($('#threshold').value)||0.5,
          prefix_padding_ms:300,
          silence_duration_ms:parseInt($('#silence').value,10)||500,
          create_response:true,
          interrupt_response:true
        }
      }},
      voice:$('#voice').value,
      overlap_policy:'barge_in_on_speech'
    };
    send({type:'session.update',session});
  }

  function onEvent(ev){
    const t=ev.type;
    if(!t) return;
    switch(t){
      case 'session.created':
        // 会话由客户端在 onopen 时主动打开；这里只记录，不再重复发 session.update
        log('会话已建立 '+((ev.session&&ev.session.id)||''));
        break;
      case 'session.updated':
        setStatus('listen','正在聆听 · 直接说话'); listening=true; break;
      case 'response.created':
        setStatus('speak','智子正在回答'); break;
      case 'response.audio.delta': case 'response.output_audio.delta': {
        const d=ev.delta||ev.audio||(ev.response&&ev.response.audio);
        if(d){
          try{ playPcm(base64ToPcm16(d)); }
          catch(e){ log('音频解码失败: '+e.message); }
        }
        break;
      }
      case 'response.audio_transcript.delta': case 'response.output_audio_transcript.delta':
      case 'response.output_text.delta':
        addMsg('bot',ev.delta||''); break;
      case 'response.audio_transcript.done': case 'response.output_audio_transcript.done':
        break;
      case 'conversation.item.input_audio_transcription.delta':
        addMsg('user',ev.delta||''); break;
      case 'conversation.item.input_audio_transcription.completed':
        if(ev.transcript && !lastUserText) addMsg('user',ev.transcript);
        break;
      case 'input_audio_buffer.speech_started':
        // 用户开口 -> 打断当前播放（barge-in）
        stopPlayback();
        setStatus('listen','正在聆听 · 直接说话');
        closeTurns();
        break;
      case 'input_audio_buffer.speech_stopped':
        setStatus('','处理中…'); break;
      case 'response.done':
        if(ev.response&&ev.response.status==='cancelled') stopPlayback();
        setTimeout(()=>{ if(started&&!activeSources.size){ setStatus('listen','正在聆听 · 直接说话'); closeTurns(); } },400);
        break;
      case 'output_audio_buffer.cleared':
        stopPlayback(); break;
      case 'error': {
        const code=ev.code||(ev.error&&ev.error.code);
        const msg=typeof ev.error==='string'?ev.error:(ev.error&&ev.error.message)||ev.message||code||'未知错误';
        log('ERROR '+code+': '+msg);
        addMsg('sys','错误：'+msg);
        setStatus('err','出错了 · 看日志');
        break;
      }
      case 'session.closed':
        if(started) addMsg('sys','会话已关闭'); break;
      default: break;
    }
    if(['session.created','session.updated','response.created','response.done','error'].includes(t))
      log('event: '+t);
  }

  function connect(){
    const proto=location.protocol==='https:'?'wss:':'ws:';
    const url=proto+'//'+location.host+'/v1/realtime?duplex=1';
    log('连接 '+url);
    ws=new WebSocket(url);
    ws.binaryType='arraybuffer';
    // duplex 协议：客户端主动发第一条 session.update 打开会话，之后服务端才回 session.created
    ws.onopen=()=>{ log('WebSocket 已连接'); setStatus('','正在初始化会话…'); sessionUpdate(); };
    ws.onmessage=e=>{
      try{ onEvent(JSON.parse(e.data)); }
      catch(err){ log('消息解析失败: '+e.data.slice(0,120)); }
    };
    ws.onerror=()=>{ log('WebSocket 错误'); };
    ws.onclose=e=>{
      log('WebSocket 关闭 code='+e.code);
      if(started){ setStatus('err','连接断开'); addMsg('sys','连接断开，请点「结束」后重试'); }
      btnStart.disabled=false; btnStart.textContent='重新连接';
    };
  }

  // ---------- 麦克风 ----------
  async function startMic(){
    if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)
      throw new Error('浏览器不支持录音，或当前不是 HTTPS/localhost');
    micStream=await navigator.mediaDevices.getUserMedia({audio:{
      echoCancellation:true, noiseSuppression:true, autoGainControl:true, channelCount:1
    }});
    audioCtx=new (window.AudioContext||window.webkitAudioContext)();
    await audioCtx.resume();
    PLAY_RATE=parseInt($('#rate').value,10)||24000;
    sinkGain=audioCtx.createGain(); sinkGain.gain.value=1.0; sinkGain.connect(audioCtx.destination);
    nextPlayTime=audioCtx.currentTime;

    const src=audioCtx.createMediaStreamSource(micStream);
    // ScriptProcessor 的缓冲区按设备采样率运行，需重采样到 24k
    procNode=audioCtx.createScriptProcessor(4096,1,1);
    procNode.onaudioprocess=e=>{
      if(!listening||!ws||ws.readyState!==1) return;
      const raw=e.inputBuffer.getChannelData(0);
      let peak=0; for(let i=0;i<raw.length;i+=8) peak=Math.max(peak,Math.abs(raw[i]));
      meterBar.style.width=Math.min(100,peak*260).toFixed(0)+'%';
      const pcm=resample(raw,audioCtx.sampleRate,REC_RATE);
      send({type:'input_audio_buffer.append',audio:pcm16ToBase64(pcm)});
    };
    const silent=audioCtx.createGain(); silent.gain.value=0;
    src.connect(procNode); procNode.connect(silent); silent.connect(audioCtx.destination);
  }
  function stopMic(){
    listening=false;
    if(procNode){ try{procNode.onaudioprocess=null;procNode.disconnect()}catch(e){} procNode=null; }
    if(micStream){ micStream.getTracks().forEach(t=>t.stop()); micStream=null; }
    if(audioCtx){ audioCtx.close().catch(()=>{}); audioCtx=null; }
    stopPlayback();
    meterBar.style.width='0';
  }

  // ---------- 控制 ----------
  async function start(){
    if(started) return;
    addMsg('sys','正在连接服务…');
    try{
      await startMic();
      started=true;
      btnStart.disabled=true; btnStop.disabled=false;
      connect();
    }catch(e){
      log('启动失败: '+e.message);
      addMsg('sys','启动失败：'+e.message);
      setStatus('err','麦克风不可用');
      stopMic();
    }
  }
  function stop(){
    started=false; listening=false;
    if(ws){ try{ws.close()}catch(e){} ws=null; }
    stopMic();
    btnStart.disabled=false; btnStop.disabled=true; btnStart.textContent='开始通话';
    setStatus('','未连接');
    addMsg('sys','通话已结束');
  }
  btnStart.addEventListener('click',start);
  btnStop.addEventListener('click',stop);
})();
</script>
</body>
</html>
"""


def guess_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("223.5.5.5", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(cert_dir: Path, ip: str):
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    cert_dir.mkdir(parents=True, exist_ok=True)
    san = f"DNS:localhost,IP:127.0.0.1,IP:{ip}"
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(cert), "-days", "365",
        "-subj", "/CN=Sophon DGX Spark", "-addext", f"subjectAltName={san}",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cert, key


async def index(request):
    # 优先读同目录下的独立 HTML（方便单独修改网页），找不到就用内嵌副本
    page_path = request.app["cfg"].page
    try:
        text = Path(page_path).expanduser().read_text(encoding="utf-8")
    except OSError:
        text = HTML
    # 新版 aiohttp 要求 charset 单独传，不能写进 content_type
    return web.Response(text=text, content_type="text/html", charset="utf-8")


async def ws_realtime(request):
    """把浏览器的 WSS 反向代理到后端 Sophon 的 WS（保留 query，如 duplex=1）。"""
    cfg = request.app["cfg"]
    query = request.rel_url.query_string
    backend = cfg.backend.rstrip("/") + "/v1/realtime" + ("?" + query if query else "")

    client = web.WebSocketResponse(autoping=True, heartbeat=30)
    await client.prepare(request)

    log = request.app["log"]
    log(f"新会话 -> {backend}")
    try:
        async with ClientSession() as sess:
            async with sess.ws_connect(backend, heartbeat=30, max_msg_size=0) as upstream:
                log("后端 WebSocket 已连接")

                async def pump(src, dst, label):
                    async for msg in src:
                        if msg.type == WSMsgType.TEXT:
                            await dst.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await dst.send_bytes(msg.data)
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR,
                                          WSMsgType.CLOSING):
                            log(f"{label} 结束: {msg.type}")
                            break

                t1 = asyncio.create_task(pump(client, upstream, "浏览器->后端"))
                t2 = asyncio.create_task(pump(upstream, client, "后端->浏览器"))
                done, pending = await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception():
                        log(f"pump 异常: {t.exception()}")
    except Exception as e:
        log(f"反代错误: {e}")
        try:
            await client.send_json({"type": "error", "error": {"code": "proxy_error", "message": str(e)}})
        except Exception:
            pass
    return client


async def on_startup(app):
    cfg = app["cfg"]
    app["log"] = lambda s: print(f"[sophon-web] {s}", flush=True)
    app["log"](f"后端: {cfg.backend}")
    scheme = "https" if cfg.self_signed else "http"
    app["log"](f"页面: {scheme}://{cfg.host}:{cfg.port}")


def main():
    ap = argparse.ArgumentParser(description="Sophon-Realtime 全双工语音网页前端")
    ap.add_argument("--backend", default="ws://127.0.0.1:8000", help="后端 WebSocket 地址")
    ap.add_argument("--port", type=int, default=8443, help="本服务端口（默认 8443）")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（Sync/Tailscale 模式用 127.0.0.1；Sync 转发连不上时试 0.0.0.0）")
    ap.add_argument("--self-signed", action="store_true",
                    help="让本脚本直接用自签 HTTPS（不用 Tailscale 时选它）")
    ap.add_argument("--cert-dir", default=str(Path.home() / "sophon" / "certs"), help="自签证书目录")
    ap.add_argument("--page", default=str(Path(__file__).with_name("sophon_frontend.html")),
                    help="网页文件路径（默认同目录的 sophon_frontend.html）")
    args = ap.parse_args()

    app = web.Application()
    app["cfg"] = args
    app.on_startup.append(on_startup)
    app.router.add_get("/", index)
    app.router.add_get("/v1/realtime", ws_realtime)

    ssl_ctx = None
    if args.self_signed:
        ip = guess_ip()
        cert, key = ensure_cert(Path(args.cert_dir).expanduser(), ip)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(str(cert), str(key))
        print(f"[sophon-web] 自签模式，证书: {cert}", flush=True)
        print(f"[sophon-web] 本机: https://localhost:{args.port}", flush=True)
        print(f"[sophon-web] 远程: https://{ip}:{args.port}（需信任自签证书）", flush=True)
    else:
        print("[sophon-web] HTTP 模式", flush=True)
        print(f"[sophon-web] 本地监听: http://{args.host}:{args.port}", flush=True)
        print("[sophon-web] NVIDIA Sync: 把远端端口映射到笔记本本地同端口，浏览器开 http://localhost:<本地端口>", flush=True)
        print("[sophon-web] 备选 Tailscale: tailscale serve --bg --https=443 http://127.0.0.1:"
              + str(args.port), flush=True)

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_ctx,
                print=None)


if __name__ == "__main__":
    main()
