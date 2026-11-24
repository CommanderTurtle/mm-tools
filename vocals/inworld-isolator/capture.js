// SPDX-License-Identifier: AGPL-3.0-only
class Capture extends AudioWorkletProcessor {
 constructor(){super();this.block=new Float32Array(Math.round(sampleRate/5));this.n=0;this.ended=false;this.port.onmessage=()=>{this.ended=true;if(this.n)this.port.postMessage(this.block.slice(0,this.n));this.port.postMessage(null)}}
 process(inputs){if(this.ended)return false;const ch=inputs[0];if(ch?.length)for(let i=0;i<ch[0].length;i++){let s=0;for(const v of ch)s+=v[i];this.block[this.n++]=s/ch.length;if(this.n===this.block.length){this.port.postMessage(this.block);this.n=0}}return true}
}
registerProcessor('capture',Capture);
