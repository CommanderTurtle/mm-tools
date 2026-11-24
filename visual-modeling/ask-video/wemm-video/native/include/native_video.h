#pragma once
#include "nlohmann/json.hpp"
#include "third_party/torch_u8_resize.h"
#include <cerrno>
#include <array>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

extern char ** environ;

namespace wemm_video {
using json = nlohmann::ordered_json;

// External FFmpeg supplies decoded RGB only. Sampling, resize, prompt layout and
// temporal grouping are C++; no shell expansion, Python or HF process is used.
class pipe_process {
    pid_t pid = -1;
    int output = -1;
public:
    pipe_process(std::vector<std::string> arguments, const std::string & log) {
        int handles[2];
        if (pipe(handles) != 0) throw std::runtime_error("Cannot create decoder pipe");
        std::vector<char *> pointers;
        for (auto & a : arguments) pointers.push_back(a.data());
        pointers.push_back(nullptr);
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions,handles[1],STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions,handles[0]);
        posix_spawn_file_actions_addclose(&actions,handles[1]);
        posix_spawn_file_actions_addopen(&actions,STDERR_FILENO,log.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0600);
        const int code=posix_spawnp(&pid,pointers[0],&actions,nullptr,pointers.data(),environ);
        posix_spawn_file_actions_destroy(&actions);
        close(handles[1]); output=handles[0];
        if (code) { close(output); output=-1; pid=-1; throw std::runtime_error("Cannot start decoder: "+std::string(std::strerror(code))); }
    }
    pipe_process(const pipe_process &)=delete;
    ~pipe_process() {
        if (output>=0) close(output);
        if (pid>0) { kill(pid,SIGTERM); int status; while(waitpid(pid,&status,0)<0 && errno==EINTR) {} }
    }
    size_t read_bytes(uint8_t * buffer, size_t size) {
        size_t used=0;
        while(used<size) {
            const ssize_t count=read(output,buffer+used,size-used);
            if (count<0 && errno==EINTR) continue;
            if (count<0) throw std::runtime_error("Decoder pipe read failed");
            if (count==0) break;
            used+=count;
        }
        return used;
    }
    std::string capture(size_t maximum=4*1024*1024) {
        std::string result; std::array<uint8_t,4096> bytes;
        for (;;) {
            const auto n=read_bytes(bytes.data(),bytes.size());
            if (!n) break;
            if (result.size()+n>maximum) throw std::runtime_error("Probe output exceeds metadata limit");
            result.append(reinterpret_cast<const char *>(bytes.data()),n);
        }
        finish();return result;
    }
    void finish() {
        int status=0;pid_t done;
        do { done=waitpid(pid,&status,0); } while(done<0 && errno==EINTR);
        pid=-1;
        if (done<0 || !WIFEXITED(status) || WEXITSTATUS(status)!=0)
            throw std::runtime_error("FFmpeg/FFprobe failed; inspect native decoder log");
    }
};

inline int round_even(double value) {
    const double low=std::floor(value), fraction=value-low;
    return int(low)+(fraction>0.5 || (fraction==0.5 && int(low)%2));
}

inline clip_image_size target_size(int width, int height, int maximum, int minimum=4096) {
    constexpr int factor=32;
    if (width<1 || height<1 || std::max(width,height)/double(std::min(width,height))>200)
        throw std::runtime_error("Video dimensions exceed the HF resize domain");
    int w=std::max(factor,round_even(width/double(factor))*factor);
    int h=std::max(factor,round_even(height/double(factor))*factor);
    if (int64_t(w)*h>maximum) {
        const double beta=std::sqrt(double(width)*height/maximum);
        w=int(std::floor(width/beta/factor))*factor;
        h=int(std::floor(height/beta/factor))*factor;
    } else if (int64_t(w)*h<minimum) {
        const double beta=std::sqrt(double(minimum)/(double(width)*height));
        w=int(std::ceil(width*beta/factor))*factor;
        h=int(std::ceil(height*beta/factor))*factor;
    }
    if (w<factor || h<factor) throw std::runtime_error("HF resize produced an invalid patch grid");
    return {w,h};
}

inline void prepare(json & request, const std::string & directory) {
    if (!request.contains("raw_video")) return;
    namespace fs = std::filesystem;
    const std::string video=fs::canonical(request.at("raw_video").get<std::string>()).string();
    pipe_process probe({"ffprobe","-v","error","-select_streams","v:0","-show_entries",
                        "stream=width,height,avg_frame_rate,nb_frames","-of","json",video},directory+"/ffprobe.log");
    auto streams=json::parse(probe.capture()).at("streams");
    if (streams.empty()) throw std::runtime_error("Video has no visual stream");
    auto info=streams.at(0);
    std::string frame_count_source="container_nb_frames";
    if (!info.contains("nb_frames") || !info.at("nb_frames").is_string() ||
        info.at("nb_frames")=="N/A" || info.at("nb_frames")=="") {
        pipe_process counter({"ffprobe","-v","error","-select_streams","v:0","-count_frames",
                              "-show_entries","stream=nb_read_frames","-of","json",video},
                             directory+"/ffprobe_count.log");
        const auto counted=json::parse(counter.capture()).at("streams");
        if (counted.empty() || !counted.at(0).contains("nb_read_frames") ||
            !counted.at(0).at("nb_read_frames").is_string() ||
            counted.at(0).at("nb_read_frames")=="N/A")
            throw std::runtime_error("FFprobe could not determine the exact decoded frame count");
        info["nb_frames"]=counted.at(0).at("nb_read_frames");
        frame_count_source="ffprobe_decoded_frames";
    }
    const int64_t total=std::stoll(info.at("nb_frames").get<std::string>());
    const std::string rate=info.at("avg_frame_rate");
    const auto slash=rate.find('/');
    if(slash==std::string::npos) throw std::runtime_error("Invalid source FPS fraction");
    const double fps=std::stod(rate.substr(0,slash))/std::stod(rate.substr(slash+1));
    if (!std::isfinite(fps)||fps<=0||total<2) throw std::runtime_error("Invalid video FPS/frame count");
    const int width=info.at("width"),height=info.at("height");
    const double target_fps=request.value("fps",2.0);
    const int max_frames=request.value("max_frames",768), min_frames=request.value("min_frames",4);
    const int total_pixels=request.value("total_pixels",25165824);
    if (!std::isfinite(target_fps)||target_fps<=0||max_frames<2||min_frames<2||total_pixels<4096)
        throw std::runtime_error("Invalid HF video sampling or pixel budget");
    const int cap=int(std::min<int64_t>(total,max_frames)/2)*2;
    const int count=int(std::min(std::max(total/fps*target_fps,double(min_frames)),double(cap)))/2*2;
    if (count<2) throw std::runtime_error("No complete video frame pair");
    std::vector<int64_t> indices;
    const double step=double(total-1)/(count-1);
    for (int i=0;i<count;++i) indices.push_back(i+1==count ? total-1 : int64_t(i*step));
    const int maximum=int(std::max(std::min(768.0*32*32,double(total_pixels)/count*2),double(int(4096*1.05))));
    const auto size=target_size(width,height,maximum);
    std::string selection="select=";
    for (size_t i=0;i<indices.size();++i) {
        if(i) selection+='+';
        selection+="eq(n\\,"+std::to_string(indices[i])+")";
    }
    pipe_process decode({"ffmpeg","-nostdin","-v","error","-threads","1","-i",video,"-an",
                         "-vf",selection,"-vsync","0","-frames:v",std::to_string(count),"-threads","1",
                         "-f","rawvideo","-pix_fmt","rgb24","pipe:1"},directory+"/decode.log");
    json prepared=json::array();
    auto media=request.at("media");
    if (media.empty() || media.back().get<std::string>()!=video)
        throw std::runtime_error("Native video must be the final explicit media item");
    for(size_t i=0;i+1<media.size();++i)
        prepared.push_back({{"path",media[i]},{"video_frame",false}});
    std::string expansion;
    clip_image_u8 original,scaled;
    original.set_size({width,height},false);
    std::vector<uint8_t> rgb(size_t(width)*height*3);
    for(int i=0;i<count;++i) {
        if(decode.read_bytes(rgb.data(),rgb.size())!=rgb.size())
            throw std::runtime_error("Source-index decoder emitted incomplete RGB frame");
        original.cpy_buf(rgb);
        wemm_torch_u8_resize::resize(original,scaled,size.width,size.height,RESIZE_ALGO_BICUBIC);
        const std::string name="frame-"+std::to_string(i)+".ppm";
        const auto path=fs::absolute(fs::path(directory)/name).string();
        std::ofstream out(path,std::ios::binary);
        out << "P6\n" << size.width << ' ' << size.height << "\n255\n";
        const auto & data=scaled.get_ro_buf();
        out.write(reinterpret_cast<const char *>(data.data()),data.size());out.close();
        if(!out) throw std::runtime_error("Cannot save native RGB frame");
        prepared.push_back({{"path",path},{"video_frame",true},{"id","hf-video+"+std::to_string(i)}});
        if(i%2==0) {
            const double seconds=(indices[i]/fps+indices[i+1]/fps)/2;
            std::ostringstream stamp;stamp.imbue(std::locale::classic());
            stamp << '<' << std::fixed << std::setprecision(1) << seconds << " seconds>";
            expansion+=stamp.str();
        }
        expansion+="<__media__>";
    }
    uint8_t extra;
    if(decode.read_bytes(&extra,1)) throw std::runtime_error("Source-index decoder emitted extra RGB bytes");
    decode.finish();
    std::string prompt=request.at("prompt");
    const auto offset=prompt.rfind("<__media__>");
    if(offset==std::string::npos) throw std::runtime_error("Missing video prompt marker");
    prompt.replace(offset,std::string("<__media__>").size(),expansion);
    request["prompt"]=prompt;
    request["native_prepared_media"]=prepared;
    request["native_input_protocol"]="hf_uniform_video_v1";
    request["native_video_metadata"]={{"path",video},{"total_num_frames",total},{"fps",fps},
        {"frame_count_source",frame_count_source},
        {"sampled_frames",count},{"source_frame_indices",indices},{"width",width},{"height",height},
        {"resized_width",size.width},{"resized_height",size.height},{"grid_thw",{count/2,size.height/16,size.width/16}},
        {"total_pixels",total_pixels},{"per_frame_max_pixels",maximum},{"decoder","FFmpeg RGB24"},
        {"resize","PyTorch-compatible uint8 antialias bicubic with per-axis int16 coefficient precision"},{"timestamp_placement","mean source indices before each pair"}};
}
} // namespace wemm_video
