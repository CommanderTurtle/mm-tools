#pragma once
#include "native_audio_encoder.h"
#include "native_audio_layout.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <chrono>

namespace wemm_audio {
struct unit {
    const mtmd_input_chunk * image=nullptr;
    std::vector<llama_token> text;
    size_t offset=0,audio_tokens=0;
    size_t size() const { return image?mtmd_input_chunk_get_n_tokens(image):audio_tokens?audio_tokens:text.size(); }
};

inline void decode_linear(llama_context * ctx,const std::vector<llama_token> & tokens,
        const float * embeddings,size_t n,int hidden,int maximum,llama_pos & past,bool last) {
    for(size_t offset=0;offset<n;) {
        const int count=std::min<size_t>(maximum,n-offset);
        std::vector<llama_pos> positions(size_t(count)*4);
        std::vector<int32_t> nseq(count,1);
        llama_seq_id id=0;std::vector<llama_seq_id *> seq(count,&id);
        std::vector<int8_t> logits(count,0);
        for(int i=0;i<count;++i) for(int axis=0;axis<3;++axis) positions[axis*count+i]=past+i;
        if(last && offset+count==n) logits.back()=1;
        llama_batch batch{count,embeddings?nullptr:const_cast<llama_token *>(tokens.data()+offset),
            embeddings?const_cast<float *>(embeddings+offset*hidden):nullptr,
            positions.data(),nseq.data(),seq.data(),logits.data()};
        need(llama_decode(ctx,batch)==0,"Native audio/text prefill failed");past+=count;offset+=count;
    }
}

inline json infer(json request,const std::string & directory,llama_model * model,llama_context * ctx,
        mtmd_context * vision,int gpu_layers,int threads,int batch_size,bool vision_flash_attention,
        std::unique_ptr<encoder> & audio,std::string & loaded_audio_model) {
    const auto started=std::chrono::steady_clock::now();
    // Optional host wall-clock diagnostics. Frozen accuracy binaries are kept
    // separately; this flag does not change tensors, batching or sampling.
    const bool profile=request.value("profile_phases",false);
    const auto clock=[&]() { return profile?std::chrono::steady_clock::now():started; };
    const auto elapsed=[&](std::chrono::steady_clock::time_point begin) {
        return std::chrono::duration<double>(clock()-begin).count();
    };
    double mel_read_seconds=0,audio_encode_seconds=0,audio_init_seconds=0;
    double visual_encode_seconds=0,language_prefill_seconds=0;
    size_t mel_bytes=0,segments_encoded=0,prefill_units=0;
    if(!request.contains("native_prepared_media")) wemm_video::prepare(request,directory);
    prepare_input(request,directory);
    need(request.contains("native_audio"),"Missing native audio preparation");
    const double preparation_seconds=elapsed(started);const auto rgb_started=clock();
    const auto * vocab=llama_model_get_vocab(model);
    const int hidden=llama_model_n_embd_inp(model);
    need(hidden>0,"Invalid WeMM language embedding size");
    mtmd::bitmaps bitmaps;
    auto options=mtmd_helper_init_opt_default();
    if(request.contains("native_prepared_media")) {
        for(const auto & item:request.at("native_prepared_media")) {
            auto b=mtmd_helper_bitmap_init_from_file(vision,item.at("path").get<std::string>().c_str(),false,options);
            need(b.bitmap && !b.video_ctx,"Native audio-video requires prepared RGB frames");
            const bool frame=item.value("video_frame",false);mtmd_bitmap_set_mergeable(b.bitmap,frame);
            if(frame) mtmd_bitmap_set_id(b.bitmap,item.at("id").get<std::string>().c_str());
            bitmaps.entries.emplace_back(b.bitmap);
        }
    } else for(const auto & path:request.at("media")) {
        auto b=mtmd_helper_bitmap_init_from_file(vision,path.get<std::string>().c_str(),false,options);
        need(b.bitmap && !b.video_ctx,"Use --video-audio for native mixed video input");
        bitmaps.entries.emplace_back(b.bitmap);
    }
    const double rgb_load_seconds=elapsed(rgb_started);const auto layout_started=clock();
    auto pointers=bitmaps.c_ptr();const std::string prompt=request.at("prompt");
    mtmd_input_text text{prompt.data(),prompt.size(),false,true};
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    need(mtmd_tokenize(vision,chunks.ptr.get(),&text,pointers.data(),pointers.size())==0,"Native audio-video tokenization failed");
    bitmaps.entries.clear();
    size_t audio_count=0;for(const auto & item:request.at("native_audio")) audio_count+=item.at("audio_tokens").get<size_t>();
    std::vector<unit> units;size_t audio_cursor=0;
    const size_t n_chunks=mtmd_input_chunks_size(chunks.ptr.get());
    for(size_t i=0;i<n_chunks;++i) {
        const auto * chunk=mtmd_input_chunks_get(chunks.ptr.get(),i);
        if(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            unit u;u.image=chunk;units.push_back(std::move(u));continue;
        }
        need(mtmd_input_chunk_get_type(chunk)==MTMD_INPUT_CHUNK_TYPE_TEXT,"Unexpected audio-video tokenizer chunk");
        size_t n;const auto * ids=mtmd_input_chunk_get_tokens_text(chunk,&n);
        std::vector<llama_token> tokens(ids,ids+n);
        if(request.value("use_audio_in_video",false) && !units.empty() && units.back().image &&
           tokens.size()>1 && tokens[0]==248054 && tokens[1]==248076) {
            // mtmd wraps the image in vision_start/end. WeMM places the audio
            // pads after the visual patches but BEFORE that same vision_end.
            size_t end=1;while(end<tokens.size() && tokens[end]==248076) ++end;
            std::rotate(tokens.begin(),tokens.begin()+1,tokens.begin()+end);
        }
        for(size_t offset=0;offset<tokens.size();) {
            const bool is_audio=tokens[offset]==248076;size_t end=offset+1;
            while(end<tokens.size() && (tokens[end]==248076)==is_audio) ++end;
            unit u;
            if(is_audio) { u.offset=audio_cursor;u.audio_tokens=end-offset;audio_cursor+=u.audio_tokens; }
            else u.text.assign(tokens.begin()+offset,tokens.begin()+end);
            units.push_back(std::move(u));offset=end;
        }
    }
    need(audio_cursor==audio_count,"Audio placeholder count differs from native encoder segments");
    std::vector<llama_token> input_ids;std::vector<int32_t> positions;
    llama_pos checked_past=0;
    for(const auto & u:units) {
        const size_t n=u.size();
        if(u.image) {
            const auto * id=mtmd_input_chunk_get_id(u.image);
            const bool video=id && std::string(id).rfind("hf-video+",0)==0;
            input_ids.insert(input_ids.end(),n,video?248057:248056);
            const auto * image=mtmd_input_chunk_get_tokens_image(u.image);
            for(size_t j=0;j<n;++j) {
                auto p=mtmd_image_tokens_get_decoder_pos(image,checked_past,j);
                positions.insert(positions.end(),{p.t,p.y,p.x});
            }
            checked_past+=mtmd_input_chunk_get_n_pos(u.image);
        } else {
            if(u.audio_tokens) input_ids.insert(input_ids.end(),n,248076);
            else input_ids.insert(input_ids.end(),u.text.begin(),u.text.end());
            for(size_t j=0;j<n;++j) positions.insert(positions.end(),3,checked_past+j);
            checked_past+=n;
        }
    }
    const int max_new=request.at("max_new_tokens");
    need(max_new>0 && input_ids.size()+max_new<=llama_n_ctx(ctx),"Audio-video context budget exceeded; input is not truncated");
    if(request.contains("expected_input_ids")) {
        need(input_ids==request.at("expected_input_ids").get<std::vector<llama_token>>(),"Native audio-video input IDs differ from frozen HF reference");
        need(positions==request.at("expected_positions").get<std::vector<int32_t>>(),"Native audio-video MRoPE positions differ from frozen HF reference");
        need(checked_past==request.at("expected_next_position").get<llama_pos>(),"Native audio-video next position differs from HF");
    }
    {
        std::ofstream out(directory+"/positions.i32",std::ios::binary);
        out.write(reinterpret_cast<const char *>(positions.data()),positions.size()*sizeof(int32_t));
        out.close();need(bool(out),"Cannot save audio-video positions");
        std::ofstream r(directory+"/native_audio_request.json");r<<request.dump()<<'\n';
        r.close();need(bool(r),"Cannot save prepared audio-video request");
    }
    const double layout_seconds=elapsed(layout_started);
    std::vector<float> embeddings;embeddings.reserve(audio_count*hidden);
    if(audio_count) {
        const std::string projector=std::filesystem::canonical(request.at("audio_projector").get<std::string>()).string();
        const std::string mode=request.value("audio_compute",std::string("bf16"));
        need(mode=="bf16" || mode=="f32","Unsupported native audio compute mode");
        if(!audio) {
            const auto begin=clock();
            audio.reset(new encoder(projector,gpu_layers!=0,threads,mode=="bf16"));loaded_audio_model=projector;
            audio_init_seconds+=elapsed(begin);
        }
        need(projector==loaded_audio_model,"Persistent worker cannot change the audio GGUF between requests");
        need(audio->embedding_size()==hidden,"Audio projector output does not match this language checkpoint");
        need(audio->bf16_rounding()==(mode=="bf16"),"Persistent worker cannot change audio compute precision between requests");
        size_t segment=0;
        for(const auto & item:request.at("native_audio")) for(const auto & path:item.at("mel_files")) {
            const auto read_started=clock();
            std::vector<float> mel(128*3000);std::ifstream in(path.get<std::string>(),std::ios::binary);
            in.read(reinterpret_cast<char *>(mel.data()),mel.size()*sizeof(float));
            char extra;need(bool(in) && !in.get(extra),"Invalid native-prepared mel size");
            mel_read_seconds+=elapsed(read_started);mel_bytes+=mel.size()*sizeof(float);
            const auto encode_started=clock();
            auto result=audio->encode(mel,request.value("save_audio_layers",false)?directory+"/audio-segment-"+std::to_string(segment):"");
            audio_encode_seconds+=elapsed(encode_started);++segments_encoded;
            embeddings.insert(embeddings.end(),result.begin(),result.end());++segment;
        }
        need(embeddings.size()==audio_count*hidden,"Native audio embedding count mismatch");
        if(request.value("save_embeddings",false)) write_floats(directory+"/audio_embeddings.f32",embeddings);
    }
    const auto reset_started=clock();llama_memory_clear(llama_get_memory(ctx),true);
    const double memory_reset_seconds=elapsed(reset_started);
    llama_pos past=0;json proof=json::array();
    for(size_t i=0;i<units.size();++i) {
        const auto & u=units[i];const size_t n=u.size();
        json row={{"kind",u.image?"image":u.audio_tokens?"audio":"text"},{"tokens",n},{"position_start",past}};
        if(u.image) {
            const auto encode_started=clock();
            need(mtmd_encode_chunk(vision,u.image)==0,"Native visual encoder failed");
            visual_encode_seconds+=elapsed(encode_started);const auto decode_started=clock();
            auto * visual=mtmd_get_output_embd(vision);need(visual,"Missing native visual embeddings");
            need(mtmd_helper_decode_image_chunk(vision,ctx,u.image,visual,past,0,batch_size,&past,nullptr,nullptr)==0,"Native visual prefill failed");
            language_prefill_seconds+=elapsed(decode_started);
        } else {
            const auto decode_started=clock();
            decode_linear(ctx,u.text,u.audio_tokens?embeddings.data()+u.offset*hidden:nullptr,n,hidden,batch_size,past,i+1==units.size());
            language_prefill_seconds+=elapsed(decode_started);
            if(u.audio_tokens) row["audio_offset"]=u.offset;
        }
        row["position_end"]=past;proof.push_back(row);++prefill_units;
    }
    need(past==checked_past,"Actual audio-video prefill positions differ from audit");
    const auto prefill_end=std::chrono::steady_clock::now();const llama_pos next=past;
    if(request.value("save_logits",false)) {
        const auto * logits=llama_get_logits_ith(ctx,-1);need(logits,"Missing first-step audio-video logits");
        write_floats(directory+"/logits.f32",std::vector<float>(logits,logits+llama_vocab_n_tokens(vocab)));
    }
    std::string response;std::vector<llama_token> generated;bool eos=false;
    for(int step=0;step<max_new;++step) {
        const float * logits=llama_get_logits_ith(ctx,-1);
        need(logits && std::all_of(logits,logits+llama_vocab_n_tokens(vocab),[](float v){return std::isfinite(v);}),"Nonfinite audio-video logits");
        const llama_token token=std::max_element(logits,logits+llama_vocab_n_tokens(vocab))-logits;
        generated.push_back(token);if(llama_vocab_is_eog(vocab,token)) { eos=true;break; }
        std::vector<char> bytes(256);int size=llama_token_to_piece(vocab,token,bytes.data(),bytes.size(),0,false);
        if(size<0) { bytes.resize(-size);size=llama_token_to_piece(vocab,token,bytes.data(),bytes.size(),0,false); }
        need(size>=0,"Cannot decode audio-video response token");response.append(bytes.data(),size);
        if(step+1<max_new) decode_linear(ctx,{token},nullptr,1,hidden,batch_size,past,true);
    }
    json result={{"status","completed"},{"text",response},{"token_ids",generated},{"finish_reason",eos?"eos":"length"},
        {"input_tokens",input_ids.size()},{"input_token_ids",input_ids},{"next_position",next},{"chunks",proof},
        {"audio_info",request.at("native_audio")},{"video_info",request.contains("native_video_metadata")?json::array({request.at("native_video_metadata")}):json::array()},
        {"input_protocol",request.value("use_audio_in_video",false)?"hf_complete_audio_video_v1":"hf_audio_timestamp_v1"},
        {"native_audio",true},{"has_audio",audio_count>0},
        {"audio_fallback",request.value("native_audio_fallback",std::string())},
        {"Python_model_or_preprocessing",false},{"audio_projector",loaded_audio_model},
        {"native_config",{{"context",llama_n_ctx(ctx)},{"batch",batch_size},{"gpu_layers",gpu_layers},{"threads",threads},
                          {"vision_flash_attention",vision_flash_attention},
                          {"audio_compute",audio && !audio->bf16_rounding()?"f32":"bf16"}}},
        {"prefill_seconds",std::chrono::duration<double>(prefill_end-started).count()},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()},
        {"runtime","C++/ggml Whisper+QFormer+vision+language; no Python/HF frontend"}};
    if(profile) result["phase_seconds"]={
        {"clock_scope","host wall time including backend synchronization; not GPU kernel profiling"},
        {"preparation",preparation_seconds},{"rgb_load",rgb_load_seconds},{"tokenize_layout_and_evidence",layout_seconds},
        {"audio_model_initialization",audio_init_seconds},{"mel_read",mel_read_seconds},
        {"audio_encode",audio_encode_seconds},{"memory_reset",memory_reset_seconds},
        {"visual_encode",visual_encode_seconds},{"language_prefill",language_prefill_seconds},
        {"generation",std::chrono::duration<double>(std::chrono::steady_clock::now()-prefill_end).count()},
        {"mel_bytes_read",mel_bytes},{"audio_segments_encoded",segments_encoded},{"prefill_units",prefill_units}};
    return result;
}
} // namespace wemm_audio
