#pragma once
#include "nlohmann/json.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "native_video.h"
#include "native_audio_layout.h"

// Named CLI arguments enter the same native engine as the retained line protocol.
// No Python interpreter, HF processor or external embeddings are used. FFmpeg
// subprocesses only decode raw RGB; native C++ handles sampling and preprocessing.
struct wemm_native_cli {
    bool enabled = false;
    bool help = false;
    bool prepare_only = false;
    std::string directory;
    std::string output;
    std::array<std::string, 10> arguments;
    std::array<char *, 10> pointers;
    std::istringstream input;

    static void usage() {
        std::cout << "Native WeMM GGUF image/video/audio inference (C++/ggml)\n"
          "  wemm-native --model MODEL.gguf --mmproj MMPROJ.gguf --prompt TEXT\n"
          "              [--image IMAGE ...] [--video VIDEO] --output RESULT.json\n"
          "              [--audio AUDIO ... | --video-audio VIDEO] [--audio-proj AUDIO.gguf]\n"
          "              [--timestamp-interval-sec 1] [--audio-compute bf16|f32]\n"
          "              [--gpu-layers 99] [--context 73728] [--threads 4]\n"
          "              [--batch 512] [--fps 2] [--max-new-tokens 512]\n"
          "  --request FILE   replay a frozen request for independent alignment checks\n"
          "  --prepare-only   prepare raw video in C++ without loading any model\n"
          "  --help           show usage without loading models\n";
    }

    void parse(int argc, char ** argv) {
        if (argc < 2 || argv[1][0] != '-') return;
        enabled = true;
        using nlohmann::ordered_json;
        namespace fs = std::filesystem;
        std::string model, mmproj, prompt, system = "You are a helpful assistant.", request_file;
        std::vector<std::string> images;
        std::string video,audio_projector,audio_compute="bf16";
        std::vector<std::string> audios;
        bool video_audio=false;
        double timestamp_interval=std::getenv("TIMESTAMP_INTERVAL_SEC")?std::stod(std::getenv("TIMESTAMP_INTERVAL_SEC")):1.0;
        int gpu = 0, threads = 4, context = 73728, batch = 512, min_tokens = 4, max_tokens = 768;
        int max_new = 512;
        bool max_new_set = false;
        int max_frames = 768, total_pixels = 25165824;
        double fps = 2;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") { help = true; return; }
            if (key == "--prepare-only") { prepare_only = true; continue; }
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + key);
            const std::string value = argv[++i];
            if      (key == "--model" || key == "-m") model = value;
            else if (key == "--mmproj") mmproj = value;
            else if (key == "--prompt" || key == "-p") prompt = value;
            else if (key == "--system") system = value;
            else if (key == "--image") images.push_back(value);
            else if (key == "--video") video = value;
            else if (key == "--video-audio") { video=value;video_audio=true; }
            else if (key == "--audio") audios.push_back(value);
            else if (key == "--audio-proj") audio_projector=value;
            else if (key == "--audio-compute") audio_compute=value;
            else if (key == "--timestamp-interval-sec") timestamp_interval=std::stod(value);
            else if (key == "--request") request_file = value;
            else if (key == "--output") output = value;
            else if (key == "--gpu-layers") gpu = std::stoi(value);
            else if (key == "--threads") threads = std::stoi(value);
            else if (key == "--context") context = std::stoi(value);
            else if (key == "--batch") batch = std::stoi(value);
            else if (key == "--image-min-tokens") min_tokens = std::stoi(value);
            else if (key == "--image-max-tokens") max_tokens = std::stoi(value);
            else if (key == "--max-new-tokens") { max_new = std::stoi(value); max_new_set = true; }
            else if (key == "--fps") fps = std::stod(value);
            else if (key == "--max-frames") max_frames = std::stoi(value);
            else if (key == "--total-pixels") total_pixels = std::stoi(value);
            else throw std::runtime_error("Unknown option " + key);
        }
        if ((!prepare_only && (model.empty() || mmproj.empty())) || output.empty())
            throw std::runtime_error("--model, --mmproj and --output are required");
        if (threads < 1 || context < 1 || batch < 1 || min_tokens < 1 || max_tokens < min_tokens || max_new < 1 || !std::isfinite(fps) || fps <= 0)
            throw std::runtime_error("Invalid context, threads, batch, FPS or token budget");
        if (request_file.empty() == prompt.empty())
            throw std::runtime_error("Supply exactly one of --prompt and --request");
        if (!request_file.empty() && (!images.empty() || !video.empty() || !audios.empty()))
            throw std::runtime_error("A frozen --request already contains its media");
        if(video_audio && !audios.empty()) throw std::runtime_error("Use --video-audio or separate --audio inputs in one request");
        if(!std::isfinite(timestamp_interval) || timestamp_interval<=0 || (audio_compute!="bf16" && audio_compute!="f32"))
            throw std::runtime_error("Invalid audio timestamp interval or compute mode");
        if (!prepare_only && (!fs::is_regular_file(model) || !fs::is_regular_file(mmproj)))
            throw std::runtime_error("Model and mmproj must be existing regular files");
        fs::path result = fs::absolute(output);
        directory = result.string() + ".evidence";
        if (fs::exists(result) || fs::exists(directory))
            throw std::runtime_error("Output or evidence directory already exists; choose a new attempt");
        ordered_json request;
        if (!request_file.empty()) {
            std::ifstream source(request_file);
            if (!source) throw std::runtime_error("Cannot open frozen request");
            source >> request;
            if (max_new_set) request["max_new_tokens"] = max_new;
        } else {
            std::string media_prompt;
            std::vector<std::string> media;
            for (const auto & path : images) {
                media.push_back(fs::canonical(path).string());
                media_prompt += "<__media__>\n";
            }
            if (!video.empty()) {
                media.push_back(fs::canonical(video).string());
                media_prompt += "<__media__>\n";
            }
            for(const auto & path:audios) { (void)path;media_prompt+="<__audio__>\n"; }
            request = {{"prompt", "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" +
                       media_prompt + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"},
                       {"media", media}, {"max_new_tokens", max_new}, {"fps", fps},
                       {"save_embeddings", false}};
            if (!video.empty()) {
                request["raw_video"] = fs::canonical(video).string();
                request["max_frames"] = max_frames;
                request["total_pixels"] = total_pixels;
            }
            if(!audios.empty() || video_audio) {
                request["raw_audio"]=ordered_json::array();
                if(video_audio) request["raw_audio"].push_back(fs::canonical(video).string());
                else for(const auto & path:audios) request["raw_audio"].push_back(fs::canonical(path).string());
                request["use_audio_in_video"]=video_audio;
                request["timestamp_interval_sec"]=timestamp_interval;
            }
        }
        if(request.contains("raw_audio")) {
            if(!audio_projector.empty()) request["audio_projector"]=fs::canonical(audio_projector).string();
            if(!prepare_only && !request.contains("audio_projector"))
                throw std::runtime_error("Audio input requires --audio-proj with WeMM audio GGUF");
            request["audio_compute"]=audio_compute;
        }
        fs::create_directories(directory);
        if(!request.contains("native_prepared_media")) wemm_video::prepare(request,directory);
        wemm_audio::prepare_input(request,directory);
        std::ofstream request_stream(directory + "/request.json");
        request_stream << request.dump() << '\n';
        request_stream.close();
        if (!request_stream) throw std::runtime_error("Cannot write native request evidence");
        arguments = {argv[0], prepare_only ? "" : fs::canonical(model).string(), prepare_only ? "" : fs::canonical(mmproj).string(),
                     std::to_string(gpu), std::to_string(threads), std::to_string(context),
                     std::to_string(batch), std::to_string(min_tokens), std::to_string(max_tokens), std::to_string(fps)};
        for (size_t i = 0; i < arguments.size(); ++i) pointers[i] = arguments[i].data();
        input.str(directory + "\n");
        output = result.string();
    }
};
