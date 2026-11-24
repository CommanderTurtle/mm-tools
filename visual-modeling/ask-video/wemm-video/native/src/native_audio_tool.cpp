#include "native_audio_encoder.h"
#include <chrono>
#include <iostream>
#include <execinfo.h>

int main(int argc,char ** argv) {
    using namespace wemm_audio;
    std::signal(SIGSEGV,[](int signum) {
        void * frames[48];const int count=backtrace(frames,48);
        backtrace_symbols_fd(frames,count,STDERR_FILENO);_exit(128+signum);
    });
    try {
        std::string source,model,output,mel_file;
        bool use_gpu=false,prepare_only=false,bf16=true;
        int threads=4;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if(key=="--help") {
                std::cout << "Native WeMM audio frontend diagnostic\n"
                  "--audio FILE --audio-proj AUDIO.gguf --output NEW_DIRECTORY [--gpu|--cpu]\n"
                  "[--prepare-only] [--threads 4] [--compute bf16|f32]\n"
                  "--reference-mel FILE uses offline features for isolated encoder comparison only.\n";
                return 0;
            }
            if(key=="--gpu") { use_gpu=true;continue; }
            if(key=="--cpu") { use_gpu=false;continue; }
            if(key=="--prepare-only") { prepare_only=true;continue; }
            need(i+1<argc,"Missing value for "+key);
            const std::string value=argv[++i];
            if(key=="--audio") source=value;
            else if(key=="--audio-proj") model=value;
            else if(key=="--output") output=value;
            else if(key=="--threads") threads=std::stoi(value);
            else if(key=="--reference-mel") mel_file=value;
            else if(key=="--compute") { need(value=="bf16"||value=="f32","Invalid compute mode");bf16=value=="bf16"; }
            else throw std::runtime_error("Unknown option "+key);
        }
        namespace fs=std::filesystem;
        need(!output.empty() && (!source.empty() != !mel_file.empty()),"Supply output and exactly one audio/reference-mel input");
        need(!fs::exists(output),"Output exists; choose a new diagnostic attempt");
        need(threads>0 && threads<=32,"Invalid thread budget");
        need(prepare_only || fs::is_regular_file(model),"Audio GGUF is required for encoder inference");
        fs::create_directories(output);
        auto start=std::chrono::steady_clock::now();
        prepared data;
        if(!source.empty()) data=preprocess(fs::canonical(source).string(),output,true,
            model.empty()?300:read_audio_protocol(model).windows);
        else {
            std::vector<float> mel(128*3000);
            std::ifstream stream(mel_file,std::ios::binary);
            stream.read(reinterpret_cast<char *>(mel.data()),mel.size()*sizeof(float));
            char extra;need(bool(stream) && !stream.get(extra),"Reference mel must be exactly 128x3000 F32");
            data.segments.push_back(std::move(mel));data.metadata={{"offline_reference_mel",mel_file}};
        }
        if(!prepare_only) {
            std::cerr<<"native_audio: loading ggml backends\n";
            ggml_backend_load_all();
            std::cerr<<"native_audio: constructing Whisper/QFormer graph\n";
            encoder native(model,use_gpu,threads,bf16);
            std::cerr<<"native_audio: graph allocated, encoding segments\n";
            for(size_t i=0;i<data.segments.size();++i)
                native.encode(data.segments[i],output+"/segment-"+std::to_string(i));
        }
        data.metadata["status"]="completed";
        data.metadata["encoder_executed"]=!prepare_only;
        data.metadata["compute"]=bf16?"bf16-operation-rounding-f32-accumulation":"f32";
        data.metadata["GPU"]=use_gpu?1:0;
        data.metadata["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream stream(output+"/completion.json");stream<<data.metadata.dump(2)<<'\n';
        stream.close();need(bool(stream),"Cannot save audio completion");
        std::cout<<data.metadata.dump()<<'\n';
        return 0;
    } catch(const std::exception & error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
