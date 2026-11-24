#pragma once
#include "native_video.h"
#include <complex>
#include <limits>

namespace wemm_audio {
using json = nlohmann::ordered_json;
constexpr int sample_rate = 16000, segment_samples = 480000;
constexpr int mel_bins = 128, mel_frames = 3000, fft_size = 400, hop = 160;

inline void need(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

inline void write_floats(const std::string & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
    out.close(); need(bool(out), "Cannot save native audio evidence: " + path);
}

// Keep the release's float decode -> libsndfile PCM16 -> float roundtrip. Direct
// ffmpeg pcm_s16le output rounds differently around quantization boundaries.
inline std::vector<float> decode(const std::string & source, const std::string & directory) {
    wemm_video::pipe_process process({"ffmpeg","-nostdin","-v","error","-threads","1",
        "-i",source,"-map","0:a:0","-vn","-ac","1","-ar","16000","-threads","1",
        "-f","f32le","-acodec","pcm_f32le","pipe:1"},directory+"/audio_decode.log");
    std::vector<float> result;
    std::array<uint8_t,65536> bytes;
    for (;;) {
        size_t n=process.read_bytes(bytes.data(),bytes.size());
        if (!n) break;
        need(n%sizeof(float)==0,"Incomplete float PCM decoder output");
        need(result.size()+n/4 <= size_t(sample_rate)*60*60*8,"Audio exceeds explicit eight-hour input limit");
        for (size_t i=0;i<n;i+=sizeof(float)) {
            float x; std::memcpy(&x,bytes.data()+i,sizeof(float));
            need(std::isfinite(x),"Nonfinite decoded audio");
            // libsndfile's default float-to-PCM16 write uses floor and clipping.
            double q=std::floor(double(x)*32768.0);
            q=std::min(32767.0,std::max(-32768.0,q));
            result.push_back(float(q/32768.0));
        }
    }
    process.finish();need(!result.empty(),"Audio has no samples");return result;
}

class logmel {
    using z = std::complex<double>;
    std::array<z,fft_size> roots;
    std::array<float,fft_size> window;
    std::array<std::array<float,201>,mel_bins> filters;

    static double to_mel(double hz) {
        return hz < 1000.0 ? hz/(200.0/3.0) : 15.0+std::log(hz/1000.0)/(std::log(6.4)/27.0);
    }
    static double to_hz(double mel) {
        return mel < 15.0 ? mel*(200.0/3.0) : 1000.0*std::exp((mel-15.0)*(std::log(6.4)/27.0));
    }
    void fft(const z * input, int stride, z * out, int n) const {
        if (n==1) { out[0]=input[0]; return; }
        const int radix=n%2==0?2:5, m=n/radix;
        std::array<z,fft_size> temporary;
        for(int r=0;r<radix;++r) fft(input+r*stride,stride*radix,temporary.data()+r*m,m);
        for(int k=0;k<n;++k) {
            z sum(0,0);
            for(int r=0;r<radix;++r) sum+=temporary[r*m+k%m]*roots[(r*k*(fft_size/n))%fft_size];
            out[k]=sum;
        }
    }
public:
    logmel() {
        const double pi=std::acos(-1.0);
        for(int i=0;i<fft_size;++i) {
            roots[i]=std::polar(1.0,-2*pi*i/fft_size);
            window[i]=float(0.5*(1-std::cos(2*pi*i/fft_size)));
        }
        std::array<double,mel_bins+2> hz;
        for(int i=0;i<mel_bins+2;++i) hz[i]=to_hz(to_mel(8000.0)*i/(mel_bins+1));
        for(int m=0;m<mel_bins;++m) for(int k=0;k<=fft_size/2;++k) {
            double frequency=double(k)*sample_rate/fft_size;
            double lower=(frequency-hz[m])/(hz[m+1]-hz[m]);
            double upper=(hz[m+2]-frequency)/(hz[m+2]-hz[m+1]);
            filters[m][k]=float(std::max(0.0,std::min(lower,upper))*2.0/(hz[m+2]-hz[m]));
        }
    }

    std::vector<float> segment(const float * input, size_t length) const {
        need(length>0 && length<=segment_samples,"Invalid audio segment length");
        std::vector<float> padded(segment_samples,0.0f);
        std::copy(input,input+length,padded.begin());
        std::vector<float> result(size_t(mel_bins)*mel_frames);
        float maximum=-std::numeric_limits<float>::infinity();
        std::array<z,fft_size> frame, spectrum;
        std::array<float,201> power;
        // STFT center=True uses reflection at both edges of the padded 30s
        // segment. The final STFT frame is dropped, leaving exactly 3000.
        for(int t=0;t<mel_frames;++t) {
            for(int i=0;i<fft_size;++i) {
                int index=t*hop+i-fft_size/2;
                if(index<0) index=-index;
                if(index>=segment_samples) index=2*segment_samples-2-index;
                frame[i]=z(float(padded[index]*window[i]),0);
            }
            fft(frame.data(),1,spectrum.data(),fft_size);
            for(int k=0;k<=200;++k) {
                // torch.stft returns complex64 before abs().pow(2).
                const float re=float(spectrum[k].real()), im=float(spectrum[k].imag());
                const float magnitude=std::hypot(re,im);
                power[k]=magnitude*magnitude;
            }
            for(int m=0;m<mel_bins;++m) {
                float sum=0;
                for(int k=0;k<=200;++k) sum+=power[k]*filters[m][k];
                float value=std::log10(std::max(sum,1e-10f));
                result[size_t(m)*mel_frames+t]=value;maximum=std::max(maximum,value);
            }
        }
        for(auto & x:result) x=(std::max(x,maximum-8.0f)+4.0f)/4.0f;
        return result;
    }
};

struct prepared {
    size_t samples=0;
    std::vector<std::vector<float>> segments;
    json metadata;
};

inline prepared preprocess(const std::string & source,const std::string & directory,bool save,int tokens_per_segment=300) {
    auto samples=decode(source,directory);
    prepared result;result.samples=samples.size();
    if(save) write_floats(directory+"/audio_pcm16.f32",samples);
    samples.resize(std::max<size_t>(sample_rate,samples.size()),0.0f);
    logmel extractor;
    for(size_t offset=0;offset<samples.size();offset+=segment_samples) {
        result.segments.push_back(extractor.segment(samples.data()+offset,std::min<size_t>(segment_samples,samples.size()-offset)));
        if(save) write_floats(directory+"/audio_mel_"+std::to_string(result.segments.size()-1)+".f32",result.segments.back());
    }
    result.metadata={{"source",source},{"sample_rate",sample_rate},{"decoded_samples",result.samples},
        {"duration",double(result.samples)/sample_rate},{"segments",result.segments.size()},
        {"tokens_per_segment",tokens_per_segment},{"audio_tokens",tokens_per_segment*result.segments.size()},
        {"protocol","release_pcm16_whisper30_qformer_v1"},{"native_audio_preprocessing",true}};
    return result;
}

inline std::vector<int64_t> timestamp_groups(int64_t count,double duration,double interval,double seconds_per_token=0.1) {
    need(count>0 && duration>0 && std::isfinite(duration) && interval>0 && std::isfinite(interval),"Invalid audio timestamp grouping");
    need(std::isfinite(seconds_per_token) && seconds_per_token>0,"Invalid checkpoint audio token duration");
    duration=std::min(duration,count*seconds_per_token);
    const int64_t effective=std::max<int64_t>(1,std::min<int64_t>(count,std::ceil(duration/seconds_per_token-1e-9)));
    const int64_t groups=std::max<int64_t>(1,std::ceil(duration*(1/interval)-1e-9));
    std::vector<int64_t> sizes(groups,effective/groups);
    const int64_t remainder=effective%groups;
    if(remainder) for(int64_t i=1;i<=remainder;++i) ++sizes[std::min<int64_t>(groups-1,std::floor(i*(double(groups)/remainder))-1)];
    sizes.back()+=count-effective;
    return sizes;
}
} // namespace wemm_audio
