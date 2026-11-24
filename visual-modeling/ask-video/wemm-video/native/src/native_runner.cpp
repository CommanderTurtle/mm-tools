#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "native_cli.h"
#include "native_audio_pipeline.h"

using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> data(256);
    int n = llama_token_to_piece(vocab, token, data.data(), data.size(), 0, false);
    if (n < 0) {
        data.resize(-n);
        n = llama_token_to_piece(vocab, token, data.data(), data.size(), 0, false);
    }
    require(n >= 0, "cannot decode output token");
    return std::string(data.data(), n);
}

int main(int argc, char ** argv) {
    wemm_native_cli cli;
    try {
        cli.parse(argc, argv);
        if (cli.help) { cli.usage(); return 0; }
        if (cli.prepare_only) {
            const auto result = json({{"status","prepared"},{"request",cli.directory+"/request.json"},
                {"runtime","C++ FFmpeg RGB decode + native sampling/resize/layout; no model loaded"}}).dump();
            std::ofstream out(cli.output);out << result << '\n';out.close();
            if (!out) throw std::runtime_error("Cannot write preparation receipt");
            std::cout << result << '\n';return 0;
        }
        if (cli.enabled) { argc = 10; argv = cli.pointers.data(); }
        if (cli.enabled) {
            // Match the measured native accuracy configuration by default.
            // Explicit environment settings remain available for controlled A/Bs.
            if (!std::getenv("GGML_CUDA_DISABLE_GRAPHS")) setenv("GGML_CUDA_DISABLE_GRAPHS","1",0);
            if (!std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")) setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE","f32",0);
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    if (argc != 10) {
        std::cerr << "usage: wemm-native MODEL MMPROJ GPU_LAYERS THREADS CONTEXT BATCH MIN_IMAGE_TOKENS MAX_IMAGE_TOKENS FPS\n";
        return 2;
    }
    try {
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = std::stoi(argv[3]);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "language model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = std::stoi(argv[5]);
        cp.n_batch = cp.n_ubatch = std::stoi(argv[6]);
        cp.n_threads = cp.n_threads_batch = std::stoi(argv[4]);
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "language context allocation failed");
        auto vp = mtmd_context_params_default();
        vp.use_gpu = mp.n_gpu_layers != 0;
        vp.n_threads = cp.n_threads;
        vp.image_min_tokens = std::stoi(argv[7]);
        vp.image_max_tokens = std::stoi(argv[8]);
        vp.warmup = false;
        vp.flash_attn_type = cli.enabled ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (const char * flag = std::getenv("WEMM_NATIVE_VISION_FLASH_ATTN")) {
            require(std::string(flag) == "0" || std::string(flag) == "1", "invalid visual attention setting");
            if (std::string(flag) == "0") vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision(
            mtmd_init_from_file(argv[2], model.get(), vp), mtmd_free);
        require(bool(vision) && mtmd_support_vision(vision.get()), "vision model load failed");
        require(mtmd_decode_use_mrope(vision.get()), "expected MRoPE vision model");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int hidden = llama_model_n_embd_inp(model.get());
        if (!cli.enabled) std::cout << "READY\n" << std::flush;
        std::string directory;
        std::unique_ptr<wemm_audio::encoder> audio;
        std::string loaded_audio_model;
        std::istream & requests = cli.enabled ? static_cast<std::istream &>(cli.input) : std::cin;
        while (std::getline(requests, directory)) {
            if (directory.empty()) break;
            try {
                const auto begin = clock_type::now();
                std::ifstream request_stream(directory + "/request.json");
                json request = json::parse(request_stream);
                if(request.contains("raw_audio")) {
                    const auto result=wemm_audio::infer(request,directory,model.get(),ctx.get(),vision.get(),
                        mp.n_gpu_layers,cp.n_threads,cp.n_batch,vp.flash_attn_type==LLAMA_FLASH_ATTN_TYPE_ENABLED,audio,loaded_audio_model);
                    const auto serialized=result.dump(-1,' ',false,json::error_handler_t::replace);
                    if(cli.enabled) {
                        std::ofstream out(cli.output);out<<serialized<<'\n';out.close();
                        require(bool(out),"Cannot save native audio CLI result");
                    }
                    std::cout<<serialized<<'\n'<<std::flush;continue;
                }
                const int max_new = request.at("max_new_tokens").get<int>();
                require(max_new > 0, "invalid output budget");
                std::string prompt = request.at("prompt").get<std::string>();
                mtmd::bitmaps bitmaps;
                std::vector<mtmd_helper::video_ptr> videos;
                std::vector<size_t> decoded_video_frames;
                auto options = mtmd_helper_init_opt_default();
                options.video_params.fps_target = request.value("fps", std::stof(argv[9]));
                require(std::isfinite(options.video_params.fps_target) && options.video_params.fps_target > 0,
                        "invalid video fps");
                // Upstream inserts timestamps after a frame, which separates
                // Qwen's adjacent two-frame groups (especially at FPS < 1).
                // Decode natively and put timestamps before complete pairs.
                const int64_t timestamp_interval_ms = request.value("timestamp_interval_ms", 1000);
                options.video_params.timestamp_interval_ms = 0;
                const std::string marker = "<__media__>";
                size_t media_cursor = 0;
                const bool canonical = request.contains("canonical_frames");
                const bool prepared = request.contains("native_prepared_media");
                require(!(canonical && prepared), "request cannot mix frozen and raw-video preparation");
                if (canonical) {
                    require(request.value("canonical_protocol", std::string()) == "hf_silent_rgb_v1",
                            "explicit canonical-frame protocol is required");
                    const auto & frames = request.at("canonical_frames");
                    require(!frames.empty() && frames.size() % 2 == 0, "canonical video requires complete frame pairs");
                    for (size_t j = 0; j < frames.size(); ++j) {
                        auto loaded = mtmd_helper_bitmap_init_from_file(vision.get(),
                            frames[j].get<std::string>().c_str(), false, options);
                        require(loaded.bitmap && !loaded.video_ctx, "canonical input must be a decoded RGB image");
                        mtmd_bitmap_set_mergeable(loaded.bitmap, true);
                        std::string id = "canonical+" + std::to_string(j);
                        mtmd_bitmap_set_id(loaded.bitmap, id.c_str());
                        bitmaps.entries.emplace_back(loaded.bitmap);
                    }
                }
                if (prepared) {
                    require(request.value("native_input_protocol",std::string())=="hf_uniform_video_v1", "missing native source-video protocol");
                    for (const auto & media : request.at("native_prepared_media")) {
                        auto loaded=mtmd_helper_bitmap_init_from_file(vision.get(),media.at("path").get<std::string>().c_str(),false,options);
                        require(loaded.bitmap && !loaded.video_ctx,"prepared native media must be RGB images");
                        const bool frame=media.value("video_frame",false);
                        mtmd_bitmap_set_mergeable(loaded.bitmap,frame);
                        if(frame) mtmd_bitmap_set_id(loaded.bitmap,media.at("id").get<std::string>().c_str());
                        bitmaps.entries.emplace_back(loaded.bitmap);
                    }
                }
                for (const auto & filename : canonical || prepared ? json::array() : request.at("media")) {
                    const size_t marker_pos = prompt.find(marker, media_cursor);
                    require(marker_pos != std::string::npos, "missing native media marker");
                    auto loaded = mtmd_helper_bitmap_init_from_file(vision.get(),
                        filename.get<std::string>().c_str(), false, options);
                    require(loaded.bitmap != nullptr, "media decode failed");
                    if (!loaded.video_ctx) {
                        bitmaps.entries.emplace_back(loaded.bitmap);
                        media_cursor = marker_pos + marker.size();
                        continue;
                    }
                    videos.emplace_back(loaded.video_ctx);
                    mtmd_bitmap_free(loaded.bitmap);
                    std::string video_prompt;
                    size_t frames = 0;
                    double next_timestamp_ms = 0;
                    const double fps = mtmd_helper_video_get_info(loaded.video_ctx).fps;
                    for (;;) {
                        mtmd_bitmap * frame = nullptr;
                        char * text_part = nullptr;
                        const int ret = mtmd_helper_video_read_next(loaded.video_ctx, &frame, &text_part);
                        require(ret != -2, "native video frame decode failed");
                        if (ret == -1) break;
                        if (text_part) {
                            video_prompt += text_part;
                            std::free(text_part);
                        }
                        if (!frame) continue;
                        const double elapsed_ms = frames / fps * 1000.0;
                        if (timestamp_interval_ms > 0 && frames % 2 == 0 && elapsed_ms + 1e-3 >= next_timestamp_ms) {
                            char timestamp[64];
                            const double seconds = elapsed_ms / 1000.0;
                            const int minutes = static_cast<int>(seconds / 60.0);
                            std::snprintf(timestamp, sizeof(timestamp), "[%dm%.2fs]", minutes, seconds - minutes * 60.0);
                            video_prompt += timestamp;
                            next_timestamp_ms = elapsed_ms + std::max<double>(timestamp_interval_ms, 2000.0 / fps);
                        }
                        video_prompt += marker;
                        bitmaps.entries.emplace_back(frame);
                        ++frames;
                    }
                    require(frames > 0, "native video has no decoded frames");
                    decoded_video_frames.push_back(frames);
                    prompt.replace(marker_pos, marker.size(), video_prompt);
                    media_cursor = marker_pos + video_prompt.size();
                }
                auto pointers = bitmaps.c_ptr();
                mtmd_input_text text{prompt.data(), prompt.size(), false, true};
                mtmd::input_chunks chunks(mtmd_input_chunks_init());
                require(mtmd_tokenize(vision.get(), chunks.ptr.get(), &text,
                                     pointers.data(), pointers.size()) == 0, "native tokenization failed");
                // Tokenization has copied/preprocessed each bitmap, just as it
                // frees its own expanded lazy bitmaps before returning.
                bitmaps.entries.clear();
                const size_t count = mtmd_input_chunks_size(chunks.ptr.get());
                size_t n_tokens = 0;
                for (size_t i = 0; i < count; ++i)
                    n_tokens += mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks.ptr.get(), i));
                require(n_tokens > 0 && n_tokens + max_new <= llama_n_ctx(ctx.get()), "context budget exceeded; input is not truncated");
                std::vector<llama_token> checked_ids;
                std::vector<int32_t> checked_positions;
                llama_pos checked_past = 0;
                for (size_t i = 0; i < count; ++i) {
                    const auto * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
                    const size_t n = mtmd_input_chunk_get_n_tokens(chunk);
                    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        size_t nt;
                        const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &nt);
                        checked_ids.insert(checked_ids.end(), ids, ids + nt);
                        for (size_t j = 0; j < nt; ++j)
                            checked_positions.insert(checked_positions.end(), 3, checked_past + j);
                    } else {
                        require(mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE, "unsupported input type");
                        const auto * id=mtmd_input_chunk_get_id(chunk);
                        const bool video=canonical || (id && std::string(id).rfind("hf-video+",0)==0);
                        checked_ids.insert(checked_ids.end(), n, video ? 248057 : 248056);
                        const auto * image = mtmd_input_chunk_get_tokens_image(chunk);
                        for (size_t j = 0; j < n; ++j) {
                            const auto pos = mtmd_image_tokens_get_decoder_pos(image, checked_past, j);
                            checked_positions.insert(checked_positions.end(), {pos.t, pos.y, pos.x});
                        }
                    }
                    checked_past += mtmd_input_chunk_get_n_pos(chunk);
                }
                if (canonical || request.contains("expected_input_ids")) {
                    require(request.at("expected_input_ids").get<std::vector<llama_token>>() == checked_ids,
                            "canonical input token IDs differ from HF; refusing inference");
                    require(request.at("expected_next_position").get<llama_pos>() == checked_past,
                            "canonical next position differs from HF");
                    require(request.at("expected_positions").get<std::vector<int32_t>>() == checked_positions,
                            "canonical MRoPE positions differ from HF");
                }
                if (canonical || prepared || request.value("save_embeddings", false)) {
                    std::ofstream out(directory + "/positions.i32", std::ios::binary);
                    out.write(reinterpret_cast<const char *>(checked_positions.data()), checked_positions.size() * sizeof(int32_t));
                    out.close(); require(bool(out), "cannot save position evidence");
                }
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                json proof = json::array();
                std::vector<llama_token> input_ids;
                llama_pos past = 0;
                for (size_t i = 0; i < count; ++i) {
                    const auto * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
                    const auto kind = mtmd_input_chunk_get_type(chunk);
                    const size_t n = mtmd_input_chunk_get_n_tokens(chunk);
                    json row = {{"kind", int(kind)}, {"tokens", n}, {"position_start", past}};
                    if (kind == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        size_t nt;
                        const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &nt);
                        input_ids.insert(input_ids.end(), ids, ids + nt);
                        require(mtmd_helper_eval_chunk_single(vision.get(), ctx.get(), chunk,
                            past, 0, cp.n_batch, i + 1 == count, &past) == 0, "text prefill failed");
                    } else {
                        require(kind == MTMD_INPUT_CHUNK_TYPE_IMAGE, "native audio is not implemented");
                        require(mtmd_encode_chunk(vision.get(), chunk) == 0, "native vision encoding failed");
                        float * embeddings = mtmd_get_output_embd(vision.get());
                        require(embeddings != nullptr && std::all_of(embeddings, embeddings + n * hidden,
                                [](float x) { return std::isfinite(x); }), "nonfinite visual embeddings");
                        const auto * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
                        const auto * media_id = mtmd_input_chunk_get_id(chunk);
                        row["media_id"] = media_id ? media_id : "";
                        auto pos = mtmd_image_tokens_get_decoder_pos(image_tokens, past, n - 1);
                        row["last_relative_position"] = {pos.t, pos.y, pos.x, pos.z};
                        if (request.value("save_embeddings", false)) {
                            const std::string name = "vision-" + std::to_string(i) + ".f32";
                            std::ofstream out(directory + "/" + name, std::ios::binary);
                            out.write(reinterpret_cast<const char *>(embeddings), n * hidden * sizeof(float));
                            require(bool(out), "cannot save visual evidence");
                            row["embedding_file"] = name;
                        }
                        // Image padding is an audit placeholder, never fed as a language token.
                        const bool video=canonical || (media_id && std::string(media_id).rfind("hf-video+",0)==0);
                        input_ids.insert(input_ids.end(), n, video ? 248057 : 248056);
                        require(mtmd_helper_decode_image_chunk(vision.get(), ctx.get(), chunk,
                            embeddings, past, 0, cp.n_batch, &past, nullptr, nullptr) == 0, "visual prefill failed");
                    }
                    row["position_end"] = past;
                    proof.push_back(row);
                }
                const auto prefill_end = clock_type::now();
                if (request.value("save_logits", false)) {
                    const auto * logits = llama_get_logits_ith(ctx.get(), -1);
                    require(logits != nullptr, "missing first-step logits");
                    std::ofstream out(directory + "/logits.f32", std::ios::binary);
                    out.write(reinterpret_cast<const char *>(logits), llama_vocab_n_tokens(vocab) * sizeof(float));
                    out.close(); require(bool(out), "cannot save logits");
                }
                const llama_pos next_position = past;
                std::string output;
                std::vector<llama_token> generated;
                bool eos = false;
                for (int step = 0; step < max_new; ++step) {
                    const float * logits = llama_get_logits_ith(ctx.get(), -1);
                    require(logits && std::all_of(logits, logits + llama_vocab_n_tokens(vocab),
                        [](float x) { return std::isfinite(x); }), "nonfinite logits");
                    llama_token token = std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits;
                    generated.push_back(token);
                    if (llama_vocab_is_eog(vocab, token)) { eos = true; break; }
                    output += piece(vocab, token);
                    if (step + 1 == max_new) break;
                    llama_pos pos[4] = {past, past, past, 0};
                    ++past;
                    int32_t ns = 1;
                    llama_seq_id sid = 0;
                    llama_seq_id * seq = &sid;
                    int8_t need_logits = 1;
                    llama_batch batch{1, &token, nullptr, pos, &ns, &seq, &need_logits};
                    require(llama_decode(ctx.get(), batch) == 0, "generation decode failed");
                }
                json result = {{"status", "completed"}, {"text", output}, {"token_ids", generated},
                    {"finish_reason", eos ? "eos" : "length"}, {"input_tokens", n_tokens},
                    {"input_token_ids", input_ids}, {"next_position", next_position},
                    {"chunks", proof}, {"runtime", "llama.cpp mtmd; no Python/HF frontend"},
                    {"prefill_seconds", std::chrono::duration<double>(prefill_end - begin).count()},
                    {"seconds", std::chrono::duration<double>(clock_type::now() - begin).count()}};
                result["video_info"] = json::array();
                result["native_config"] = {{"context",cp.n_ctx},{"batch",cp.n_batch},
                    {"gpu_layers",mp.n_gpu_layers},{"threads",cp.n_threads},
                    {"vision_flash_attention",vp.flash_attn_type==LLAMA_FLASH_ATTN_TYPE_ENABLED}};
                result["input_protocol"] = canonical ? "hf_silent_rgb_v1" : prepared ? "hf_uniform_video_v1" : "native_ffmpeg_v1";
                if(prepared) result["video_info"].push_back(request.at("native_video_metadata"));
                for (size_t i = 0; i < videos.size(); ++i) {
                    const auto & video = videos[i];
                    const auto info = mtmd_helper_video_get_info(video.get());
                    result["video_info"].push_back({{"fps", info.fps}, {"estimated_frames", info.n_frames},
                        {"decoded_frames", decoded_video_frames[i]}, {"timestamp_placement", "before_frame_pair"},
                        {"width", info.width}, {"height", info.height}});
                }
                // A token budget can end within a UTF-8 sequence; JSON replaces only invalid trailing bytes.
                const auto serialized = result.dump(-1, ' ', false, json::error_handler_t::replace);
                if (cli.enabled) {
                    std::ofstream out(cli.output);
                    out << serialized << '\n';
                    out.close(); require(bool(out), "cannot save native CLI result");
                }
                std::cout << serialized << '\n' << std::flush;
            } catch (const std::exception & error) {
                std::cerr << "request failed: " << error.what() << '\n';
                std::cout << json({{"error", error.what()}}).dump() << '\n' << std::flush;
                if (cli.enabled) return 1;
            }
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    llama_backend_free();
    return 0;
}
