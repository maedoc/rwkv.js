#include <vector>
#include <string>
#include <emscripten/bind.h>
#include "rwkv.cpp/rwkv.h"

using namespace emscripten;

struct RwkvCtx {
    struct rwkv_context *ctx;
    float *state_in, *state_out;
    float *logits_out;
    size_t n_vocab;

    RwkvCtx(const char *model_file_path, const uint32_t n_threads) {
        ctx = rwkv_init_from_file(model_file_path, n_threads, 0);
        n_vocab = rwkv_get_n_vocab(ctx);
        state_in = new float[rwkv_get_state_len(ctx)];
        state_out = new float[rwkv_get_state_len(ctx)];
        logits_out = new float[rwkv_get_logits_len(ctx)];
        // the graph will be built and cached, but not executed
        rwkv_eval_sequence_in_chunks(ctx, NULL, 0, 0, NULL, NULL, NULL);
    }

    ~RwkvCtx() {
        delete state_in;
        delete state_out;
        delete logits_out;
        rwkv_free(ctx);
    }

    typedef std::vector<uint32_t> token_seq;

    bool eval_seq(const token_seq tokens) {
        return rwkv_eval_sequence(ctx, tokens.data(), tokens.size(), state_in, state_out, logits_out);
    }

    std::string get_sysinfo() const {
        return std::string(rwkv_get_system_info_string());
    }
};


// Binding code
EMSCRIPTEN_BINDINGS(rwkv) {
  register_vector<uint32_t>("VectorUInt32");
  class_<RwkvCtx>("RwkvCtx")
    .constructor<const char *, const uint32_t>()
    .function("eval_seq", &RwkvCtx::eval_seq)
    .property("n_vocab", &RwkvCtx::n_vocab)
    .property("sysinfo", &RwkvCtx::get_sysinfo)
    ;

}