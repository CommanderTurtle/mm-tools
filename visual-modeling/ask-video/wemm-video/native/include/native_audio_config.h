#pragma once
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace wemm_audio {
struct audio_protocol {
    int window_frames=0,stride_frames=0,windows=0;
    double seconds_per_token=0;
};

inline audio_protocol protocol_from_config(const nlohmann::ordered_json & config) {
    if(config.at("num_mel_bins")!=128 || config.at("max_source_positions")!=1500 ||
       config.at("qformer_config").at("query_length")!=1)
        throw std::runtime_error("Unsupported Whisper30/QFormer input protocol");
    const double window=config.at("second_per_window"),stride=config.at("second_stride");
    if(!std::isfinite(window) || !std::isfinite(stride) || window<=0 || stride<=0 || window>30 || stride>30)
        throw std::runtime_error("Invalid checkpoint audio window/stride");
    audio_protocol p;
    // Match round(1500 * seconds / 30) in this checkpoint's Whisper encoder.
    p.window_frames=int(std::nearbyint(1500*window/30));
    p.stride_frames=int(std::nearbyint(1500*stride/30));
    if(p.window_frames<1 || p.window_frames>1500 || p.stride_frames<1)
        throw std::runtime_error("Invalid native QFormer frame window");
    p.windows=(1500-p.window_frames)/p.stride_frames+1;
    p.seconds_per_token=stride;
    return p;
}

inline audio_protocol read_audio_protocol(const std::string & path) {
    auto * file=gguf_init_from_file(path.c_str(),{true,nullptr});
    if(!file) throw std::runtime_error("Cannot read checkpoint audio GGUF metadata");
    const auto key=gguf_find_key(file,"wemm.audio.config");
    if(key<0) {gguf_free(file);throw std::runtime_error("Missing checkpoint audio configuration");}
    const std::string text=gguf_get_val_str(file,key);gguf_free(file);
    return protocol_from_config(nlohmann::ordered_json::parse(text));
}
}
