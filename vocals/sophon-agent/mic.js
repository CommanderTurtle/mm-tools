// SPDX-License-Identifier: AGPL-3.0-only
class Mic extends AudioWorkletProcessor{constructor(){super();this.buf=new Float32Array(1280);this.n=0}process(input){const a=input[0]?.[0];if(a)for(const x of a){this.buf[this.n++]=x;if(this.n===1280){this.port.postMessage(this.buf);this.n=0}}return true}}registerProcessor('sophon-mic',Mic);
