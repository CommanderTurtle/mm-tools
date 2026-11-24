#pragma once
#include "native_audio_features.h"
#include "native_audio_config.h"

namespace wemm_audio {
inline std::string timestamp(double seconds) {
    std::ostringstream s;s.imbue(std::locale::classic());
    s << '<' << std::fixed << std::setprecision(1) << seconds << " seconds>";
    return s.str();
}

inline void prepare_input(json & request,const std::string & directory) {
    if(!request.contains("raw_audio") || request.contains("native_audio")) return;
    need(request.contains("audio_projector"),"Audio preparation requires this checkpoint's --audio-proj GGUF");
    const auto protocol=read_audio_protocol(request.at("audio_projector").get<std::string>());
    namespace fs=std::filesystem;
    const bool av=request.value("use_audio_in_video",false);
    const auto sources=request.at("raw_audio");
    const auto cached=request.value("native_audio_cache",json::array());
    need(sources.is_array() && !sources.empty(),"raw_audio must be a nonempty ordered array");
    need(cached.empty() || cached.size()==sources.size(),"Cached native audio/source count mismatch");
    need(!av || (sources.size()==1 && request.contains("native_video_metadata")),"Audio-video input needs exactly one prepared video and audio track");
    std::string prompt=request.at("prompt");
    json audio=json::array();
    for(size_t index=0;index<sources.size();++index) {
        const auto source=fs::canonical(sources[index].get<std::string>()).string();
        const auto folder=directory+"/audio-"+std::to_string(index);
        need(!fs::exists(folder),"Native audio evidence directory exists; choose a fresh attempt");
        fs::create_directory(folder);
        if(av) {
            wemm_video::pipe_process probe({"ffprobe","-v","error","-select_streams","a:0",
                "-show_entries","stream=index","-of","json",source},folder+"/audio_probe.log");
            if(json::parse(probe.capture()).at("streams").empty()) {
                request["native_audio_fallback"]="no audio stream; video only";
                request["native_audio"]=json::array();
                request["native_audio_protocol"]="release_pcm16_whisper30_qformer_v1";
                request["native_audio_layout_version"]="torchcodec_fps6_v2";
                return;
            }
        }
        json info;
        if(cached.empty()) {
            auto data=preprocess(source,folder,true,protocol.windows);
            info=data.metadata;info["mel_files"]=json::array();
            for(size_t segment=0;segment<data.segments.size();++segment)
                info["mel_files"].push_back(fs::absolute(folder+"/audio_mel_"+std::to_string(segment)+".f32").string());
        } else {
            info=cached[index];
            need(info.at("protocol")=="release_pcm16_whisper30_qformer_v1" &&
                 fs::canonical(info.at("source").get<std::string>()).string()==source,
                 "Native audio cache source or preprocessing protocol differs");
            const auto samples=info.at("decoded_samples").get<size_t>();
            need(samples>0 && samples<=size_t(sample_rate)*60*60*8,"Invalid cached audio length");
            const size_t segments=(std::max<size_t>(sample_rate,samples)+segment_samples-1)/segment_samples;
            need(info.at("segments")==segments && info.at("audio_tokens")==segments*protocol.windows &&
                 info.at("mel_files").size()==segments,"Invalid cached native audio segment metadata");
            for(const auto & path:info.at("mel_files"))
                need(fs::is_regular_file(path.get<std::string>()) && fs::file_size(path.get<std::string>())==size_t(128)*3000*4,
                     "Invalid native-produced audio cache file");
        }
        const int64_t count=info.at("audio_tokens");
        info["checkpoint_seconds_per_token"]=protocol.seconds_per_token;
        info["qformer_window_frames"]=protocol.window_frames;info["qformer_stride_frames"]=protocol.stride_frames;
        const double duration=double(info.at("decoded_samples").get<size_t>())/sample_rate;
        std::string expansion;
        std::vector<int64_t> groups;
        if(av) {
            const auto & video=request.at("native_video_metadata");
            const auto indices=video.at("source_frame_indices").get<std::vector<int64_t>>();
            const double source_fps=video.at("fps");
            // Pinned TorchCodec 0.10.0 serializes averageFps through
            // std::to_string (six decimals). This can flip a .05s timestamp.
            std::ostringstream fps_text;fps_text.imbue(std::locale::classic());
            fps_text<<std::fixed<<std::setprecision(6)<<source_fps;
            const double fps=std::stod(fps_text.str());
            info["timestamp_fps"]=fps;
            const int pairs=indices.size()/2;
            const double sampled_fps=indices.size()/duration;
            std::string old;
            int64_t previous=0;
            for(int pair=0;pair<pairs;++pair) {
                old+=timestamp((indices[pair*2]/source_fps+indices[pair*2+1]/source_fps)/2)+"<__media__><__media__>";
                // Match torch.arange(float32) * Python scalar, then floor.
                const float scale=float(2.0/sampled_fps/protocol.seconds_per_token);
                int64_t boundary=pair+1==pairs?count:int64_t(std::floor(float(pair+1)*scale));
                boundary=std::clamp<int64_t>(boundary,0,count);
                groups.push_back(boundary-previous);previous=boundary;
                expansion+=timestamp((indices[pair*2]+indices[pair*2+1])/(2*fps))+"<__media__><__media__>";
                for(int64_t i=0;i<groups.back();++i) expansion+="<|audio_pad|>";
            }
            const auto start=prompt.find(old);
            need(start!=std::string::npos && prompt.find(old,start+1)==std::string::npos,"Cannot identify unique native audio-video prompt expansion");
            prompt.replace(start,old.size(),expansion);
            info["layout"]="audio pads inside vision start/end after each visual frame pair";
        } else {
            const double interval=request.value("timestamp_interval_sec",1.0);
            groups=timestamp_groups(count,duration,interval,protocol.seconds_per_token);
            expansion="<|audio_start|>";
            for(size_t group=0;group<groups.size();++group) {
                expansion+=timestamp(group/(1.0/interval));
                for(int64_t i=0;i<groups[group];++i) expansion+="<|audio_pad|>";
            }
            expansion+="<|audio_end|>";
            const std::string marker="<__audio__>";
            const auto start=prompt.find(marker);need(start!=std::string::npos,"Missing native audio placeholder");
            prompt.replace(start,marker.size(),expansion);
            info["timestamp_interval_sec"]=interval;info["layout"]="release balanced effective tokens and padded tail";
        }
        info["audio_groups"]=groups;audio.push_back(info);
    }
    need(prompt.find("<__audio__>")==std::string::npos,"Extra audio placeholder");
    request["prompt"]=prompt;request["native_audio"]=audio;
    request["native_audio_protocol"]="release_pcm16_whisper30_qformer_v1";
    request["native_audio_layout_version"]="torchcodec_fps6_v2";
}
} // namespace wemm_audio
