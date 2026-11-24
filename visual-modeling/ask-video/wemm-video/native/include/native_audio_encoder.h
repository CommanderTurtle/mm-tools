#pragma once
#include "native_audio_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "native_audio_config.h"
#include <memory>
#include <unordered_map>

namespace wemm_audio {
// Release-specific Whisper + two-layer BERT QFormer. All graph operations and
// model weights are native ggml. BF16 mode rounds at PyTorch operation boundaries
// while retaining F32 accumulation; it is independent of the LLM's quantization.
class encoder {
    gguf_context * file=nullptr;
    ggml_context * weights=nullptr;
    ggml_backend_buffer_t weight_buffer=nullptr;
    ggml_backend_t gpu=nullptr, cpu=nullptr;
    ggml_backend_sched_t scheduler=nullptr;
    ggml_context * graph_context=nullptr;
    ggml_cgraph * graph=nullptr;
    ggml_tensor * input=nullptr,*output=nullptr;
    std::unordered_map<std::string,ggml_tensor *> stages;
    bool bf16;
    int audio_dim=0, encoder_layers=0, encoder_heads=0;
    int qformer_dim=0, qformer_layers=0, qformer_heads=0, output_dim=0;
    audio_protocol protocol;

    ggml_tensor * w(const std::string & name) {
        std::string compact=name;
        const std::string prefix="qformer.bert.";
        if(compact.rfind(prefix,0)==0) compact.replace(0,prefix.size(),"qf.");
        auto * result=ggml_get_tensor(weights,("a."+compact).c_str());
        need(result!=nullptr,"Missing audio weight: "+name);return result;
    }
    ggml_tensor * rounded(ggml_tensor * x) {
        return bf16 ? ggml_cast(graph_context,ggml_cast(graph_context,x,GGML_TYPE_BF16),GGML_TYPE_F32) : x;
    }
    ggml_tensor * matmul(ggml_tensor * a,ggml_tensor * b) {
        auto * result=ggml_mul_mat(graph_context,a,b);
        ggml_prec_set_acc(result,GGML_PREC_F32);
        ggml_prec_set_src(result,GGML_PREC_F32,1);
        return result;
    }
    ggml_tensor * linear(ggml_tensor * x,const std::string & name,bool bias=true) {
        auto * result=matmul(w(name+".weight"),x);
        if(bias) result=ggml_add(graph_context,result,w(name+".bias"));
        return rounded(result);
    }
    ggml_tensor * norm(ggml_tensor * x,const std::string & name,float epsilon) {
        auto * result=ggml_norm(graph_context,x,epsilon);
        result=ggml_mul(graph_context,result,w(name+".weight"));
        return rounded(ggml_add(graph_context,result,w(name+".bias")));
    }
    ggml_tensor * attention(ggml_tensor * q,ggml_tensor * k,ggml_tensor * v,
                            int heads,bool pre_scale) {
        auto * c=graph_context;
        const int64_t dim=q->ne[0],nq=q->ne[1],nk=k->ne[1],batch=q->ne[2],dh=dim/heads;
        if(pre_scale) q=rounded(ggml_scale(c,q,1.0f/std::sqrt(float(dh))));
        q=ggml_permute(c,ggml_reshape_4d(c,q,dh,heads,nq,batch),0,2,1,3);
        k=ggml_permute(c,ggml_reshape_4d(c,k,dh,heads,nk,batch),0,2,1,3);
        v=ggml_cont(c,ggml_permute(c,ggml_reshape_4d(c,v,dh,heads,nk,batch),1,2,0,3));
        auto * scores=rounded(matmul(k,q));
        if(!pre_scale) scores=rounded(ggml_scale(c,scores,1.0f/std::sqrt(float(dh))));
        auto * probabilities=rounded(ggml_soft_max(c,scores));
        auto * context=rounded(matmul(v,probabilities));
        context=ggml_cont(c,ggml_permute(c,context,0,2,1,3));
        return ggml_reshape_3d(c,context,dim,nq,batch);
    }
    void stage(ggml_tensor * tensor,const std::string & name) {
        ggml_set_name(tensor,name.c_str());ggml_set_output(tensor);stages[name]=tensor;
    }
    void build() {
        ggml_init_params p{ggml_tensor_overhead()*8192+ggml_graph_overhead_custom(8192,false),nullptr,true};
        graph_context=ggml_init(p);need(graph_context,"Cannot allocate audio graph metadata");
        auto * c=graph_context;graph=ggml_new_graph_custom(c,8192,false);
        input=ggml_new_tensor_2d(c,GGML_TYPE_F32,3000,128);
        ggml_set_name(input,"audio_mel");ggml_set_input(input);
        auto * cur=rounded(input);
        for(int conv=1;conv<=2;++conv) {
            const std::string name="conv"+std::to_string(conv);
            cur=ggml_conv_1d_ph(c,w(name+".weight"),cur,conv==1?1:2,1);
            cur=rounded(ggml_add(c,cur,w(name+".bias")));
            cur=rounded(ggml_gelu_erf(c,cur));
        }
        cur=ggml_cont(c,ggml_transpose(c,cur));
        stage(cur,"conv_out");
        cur=rounded(ggml_add(c,cur,ggml_cast(c,w("embed_positions.weight"),GGML_TYPE_F32)));
        for(int layer=0;layer<encoder_layers;++layer) {
            const auto prefix="layers."+std::to_string(layer);
            auto * residual=cur;
            auto * normalized=norm(cur,prefix+".self_attn_layer_norm",1e-5f);
            auto * q=linear(normalized,prefix+".self_attn.q_proj");
            auto * k=linear(normalized,prefix+".self_attn.k_proj",false);
            auto * v=linear(normalized,prefix+".self_attn.v_proj");
            auto * attn=attention(q,k,v,encoder_heads,true);
            cur=rounded(ggml_add(c,residual,linear(attn,prefix+".self_attn.out_proj")));
            residual=cur;
            cur=norm(cur,prefix+".final_layer_norm",1e-5f);
            cur=rounded(ggml_gelu_erf(c,linear(cur,prefix+".fc1")));
            cur=rounded(ggml_add(c,residual,linear(cur,prefix+".fc2")));
            if(layer==0 || layer==encoder_layers/2-1 || layer==encoder_layers-1) stage(cur,"encoder_layer_"+std::to_string(layer));
        }
        cur=norm(cur,"layer_norm",1e-5f);stage(cur,"whisper_out");
        cur=ggml_cont(c,cur);
        auto * speech=ggml_cont(c,ggml_view_3d(c,cur,audio_dim,protocol.window_frames,protocol.windows,
            cur->nb[1],cur->nb[1]*protocol.stride_frames,0));
        auto * query=ggml_repeat_4d(c,ggml_cast(c,w("q_tokens"),GGML_TYPE_F32),qformer_dim,1,protocol.windows,1);
        query=norm(query,"qformer.bert.embeddings.LayerNorm",1e-12f);
        for(int layer=0;layer<qformer_layers;++layer) {
            const std::string prefix="qformer.bert.encoder.layer."+std::to_string(layer);
            for(const std::string kind:{"attention","crossattention"}) {
                auto * residual=query;
                auto * context=kind=="attention"?query:speech;
                auto * q=linear(query,prefix+"."+kind+".self.query");
                auto * k=linear(context,prefix+"."+kind+".self.key");
                auto * v=linear(context,prefix+"."+kind+".self.value");
                query=attention(q,k,v,qformer_heads,false);
                query=linear(query,prefix+"."+kind+".output.dense");
                query=norm(rounded(ggml_add(c,query,residual)),prefix+"."+kind+".output.LayerNorm",1e-12f);
            }
            auto * residual=query;
            query=rounded(ggml_gelu_erf(c,linear(query,prefix+".intermediate_query.dense")));
            query=linear(query,prefix+".output_query.dense");
            query=norm(rounded(ggml_add(c,query,residual)),prefix+".output_query.LayerNorm",1e-12f);
            stage(query,"qformer_layer_"+std::to_string(layer));
        }
        output=ggml_cont(c,ggml_reshape_2d(c,linear(query,"audio_proj"),output_dim,protocol.windows));
        stage(output,"audio_embeddings");ggml_build_forward_expand(graph,output);
        need(ggml_backend_sched_alloc_graph(scheduler,graph),"Cannot allocate native audio graph");
    }
public:
    encoder(const std::string & path,bool use_gpu,int threads,bool emulate_bf16=true):bf16(emulate_bf16) {
        cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        need(cpu,"No native CPU backend");ggml_backend_cpu_set_n_threads(cpu,threads);
        if(use_gpu) {
            gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
            need(gpu,"GPU audio requested but no native GPU backend");
        }
        file=gguf_init_from_file(path.c_str(),{true,&weights});need(file && weights,"Cannot read audio GGUF");
        const auto arch=gguf_find_key(file,"general.architecture");
        need(arch>=0 && std::string(gguf_get_val_str(file,arch))=="wemm_audio","Unexpected audio GGUF architecture");
        need(gguf_get_n_tensors(file)>0,"Empty audio GGUF");
        const auto config_key=gguf_find_key(file,"wemm.audio.config");
        need(config_key>=0,"Missing WeMM audio configuration");
        const auto config=json::parse(gguf_get_val_str(file,config_key));
        protocol=protocol_from_config(config);
        const auto qformer=config.at("qformer_config");
        audio_dim=config.at("d_model").get<int>();encoder_layers=config.at("encoder_layers").get<int>();
        encoder_heads=config.at("encoder_attention_heads").get<int>();
        qformer_dim=qformer.at("hidden_size").get<int>();qformer_layers=qformer.at("num_hidden_layers").get<int>();
        qformer_heads=qformer.at("num_attention_heads").get<int>();output_dim=qformer.at("out_hidden_size").get<int>();
        need(audio_dim>0 && encoder_layers>0 && encoder_heads>0 && audio_dim%encoder_heads==0 &&
             qformer_dim>0 && qformer_layers>0 && qformer_heads>0 && qformer_dim%qformer_heads==0 && output_dim>0 &&
             config.at("num_mel_bins")==128 &&
             config.at("max_source_positions")==1500 && qformer.at("query_length")==1,
             "Unsupported Whisper/QFormer architecture in audio GGUF");
        weight_buffer=ggml_backend_alloc_ctx_tensors(weights,gpu?gpu:cpu);
        need(weight_buffer,"Cannot allocate native audio weights");
        ggml_backend_buffer_set_usage(weight_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::ifstream stream(path,std::ios::binary);
        std::vector<char> buffer;
        for(int64_t i=0;i<gguf_get_n_tensors(file);++i) {
            auto * tensor=ggml_get_tensor(weights,gguf_get_tensor_name(file,i));
            const size_t n=ggml_nbytes(tensor);buffer.resize(n);
            stream.seekg(gguf_get_data_offset(file)+gguf_get_tensor_offset(file,i));
            stream.read(buffer.data(),n);need(bool(stream),"Truncated audio GGUF tensor");
            ggml_backend_tensor_set(tensor,buffer.data(),0,n);
        }
        ggml_backend_t backends[]={gpu?gpu:cpu,cpu};
        scheduler=ggml_backend_sched_new(backends,nullptr,gpu?2:1,8192,false,true);
        need(scheduler,"Cannot create native audio scheduler");build();
    }
    encoder(const encoder &)=delete;
    bool bf16_rounding() const { return bf16; }
    int embedding_size() const { return output_dim; }
    ~encoder() {
        if(scheduler) ggml_backend_sched_free(scheduler);
        if(graph_context) ggml_free(graph_context);
        if(weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if(weights) ggml_free(weights);
        if(file) gguf_free(file);
        if(gpu) ggml_backend_free(gpu);
        if(cpu) ggml_backend_free(cpu);
    }
    std::vector<float> encode(const std::vector<float> & mel,const std::string & diagnostic="") {
        need(mel.size()==size_t(128)*3000,"Audio encoder requires 128x3000 mel segment");
        ggml_backend_tensor_set(input,mel.data(),0,mel.size()*sizeof(float));
        need(ggml_backend_sched_graph_compute(scheduler,graph)==GGML_STATUS_SUCCESS,"Native audio graph compute failed");
        ggml_backend_sched_synchronize(scheduler);
        std::vector<float> result(output_dim*protocol.windows);
        ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
        need(std::all_of(result.begin(),result.end(),[](float v){return std::isfinite(v);}),"Nonfinite native audio embeddings");
        if(!diagnostic.empty()) for(const auto & stage:stages) {
            std::vector<float> data(ggml_nelements(stage.second));
            ggml_backend_tensor_get(stage.second,data.data(),0,data.size()*sizeof(float));
            write_floats(diagnostic+"-"+stage.first+".f32",data);
        }
        return result;
    }
};
} // namespace wemm_audio
