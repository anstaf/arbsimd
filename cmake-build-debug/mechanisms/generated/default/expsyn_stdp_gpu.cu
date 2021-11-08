#include <arbor/gpu/gpu_common.hpp>
#include <arbor/gpu/math_cu.hpp>
#include <arbor/gpu/reduce_by_key.hpp>
#include <arbor/mechanism_abi.h>

namespace arb {
namespace default_catalogue {

#define PPACK_IFACE_BLOCK \
auto  _pp_var_width             __attribute__((unused)) = params_.width;\
auto  _pp_var_n_detectors       __attribute__((unused)) = params_.n_detectors;\
auto* _pp_var_vec_ci            __attribute__((unused)) = params_.vec_ci;\
auto* _pp_var_vec_di            __attribute__((unused)) = params_.vec_di;\
auto* _pp_var_vec_t             __attribute__((unused)) = params_.vec_t;\
auto* _pp_var_vec_dt            __attribute__((unused)) = params_.vec_dt;\
auto* _pp_var_vec_v             __attribute__((unused)) = params_.vec_v;\
auto* _pp_var_vec_i             __attribute__((unused)) = params_.vec_i;\
auto* _pp_var_vec_g             __attribute__((unused)) = params_.vec_g;\
auto* _pp_var_temperature_degC  __attribute__((unused)) = params_.temperature_degC;\
auto* _pp_var_diam_um           __attribute__((unused)) = params_.diam_um;\
auto* _pp_var_time_since_spike  __attribute__((unused)) = params_.time_since_spike;\
auto* _pp_var_node_index        __attribute__((unused)) = params_.node_index;\
auto* _pp_var_peer_index        __attribute__((unused)) = params_.peer_index;\
auto* _pp_var_multiplicity      __attribute__((unused)) = params_.multiplicity;\
auto* _pp_var_state_vars        __attribute__((unused)) = params_.state_vars;\
auto* _pp_var_weight            __attribute__((unused)) = params_.weight;\
auto& _pp_var_events            __attribute__((unused)) = params_.events;\
auto& _pp_var_mechanism_id      __attribute__((unused)) = params_.mechanism_id;\
auto& _pp_var_index_constraints __attribute__((unused)) = params_.index_constraints;\
auto* _pp_var_g __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_apre __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_apost __attribute__((unused)) = params_.state_vars[2];\
auto* _pp_var_weight_plastic __attribute__((unused)) = params_.state_vars[3];\
auto* _pp_var_tau __attribute__((unused)) = params_.parameters[0];\
auto* _pp_var_taupre __attribute__((unused)) = params_.parameters[1];\
auto* _pp_var_taupost __attribute__((unused)) = params_.parameters[2];\
auto* _pp_var_Apre __attribute__((unused)) = params_.parameters[3];\
auto* _pp_var_Apost __attribute__((unused)) = params_.parameters[4];\
auto* _pp_var_e __attribute__((unused)) = params_.parameters[5];\
auto* _pp_var_max_weight __attribute__((unused)) = params_.parameters[6];\
//End of IFACEBLOCK

namespace {

using ::arb::gpu::exprelr;
using ::arb::gpu::safeinv;
using ::arb::gpu::min;
using ::arb::gpu::max;

__global__
void init(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        _pp_var_g[tid_] =  0.;
        _pp_var_apre[tid_] =  0.;
        _pp_var_apost[tid_] =  0.;
        _pp_var_weight_plastic[tid_] =  0.;
    }
}

__global__
void multiply(arb_mechanism_ppack params_) {
    PPACK_IFACE_BLOCK;
    auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    auto idx_ = blockIdx.y;    if(tid_<_pp_var_width) {
        _pp_var_state_vars[idx_][tid_] *= _pp_var_multiplicity[tid_];
    }
}

__global__
void advance_state(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type a_1_, a_0_, a_2_, ll0_, ll1_, ll2_, ll4_, ll3_, ll5_;
        ll5_ =  0.;
        ll4_ =  0.;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        a_0_ =  -1.0/_pp_var_tau[tid_];
        ll0_ = a_0_*dt;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_g[tid_] = _pp_var_g[tid_]*ll1_;
        a_1_ =  -1.0/_pp_var_taupre[tid_];
        ll2_ = a_1_*dt;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_apre[tid_] = _pp_var_apre[tid_]*ll3_;
        a_2_ =  -1.0/_pp_var_taupost[tid_];
        ll4_ = a_2_*dt;
        ll5_ = ( 1.0+ 0.5*ll4_)/( 1.0- 0.5*ll4_);
        _pp_var_apost[tid_] = _pp_var_apost[tid_]*ll5_;
    }
}

__global__
void compute_currents(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned lane_mask_ = arb::gpu::ballot(0xffffffff, tid_<n_);
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type conductivity_ = 0;
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type current_ = 0;
        arb_value_type i = 0;
        i = _pp_var_g[tid_]*(v-_pp_var_e[tid_]);
        current_ = i;
        conductivity_ = _pp_var_g[tid_];
        ::arb::gpu::reduce_by_key(_pp_var_weight[tid_]*conductivity_,_pp_var_vec_g, node_indexi_, lane_mask_);
        ::arb::gpu::reduce_by_key(_pp_var_weight[tid_]*current_,_pp_var_vec_i, node_indexi_, lane_mask_);
    }
}

__global__
void apply_events(arb_mechanism_ppack params_, arb_deliverable_event_stream stream) {
    PPACK_IFACE_BLOCK;
    auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid_<stream.n_streams) {
        auto begin = stream.events + stream.begin[tid_];
        auto end   = stream.events + stream.end[tid_];
        for (auto p = begin; p<end; ++p) {
            if (p->mech_id==_pp_var_mechanism_id) {
                auto tid_ = p->mech_index;
                auto weight = p->weight;
                _pp_var_g[tid_] = max( 0., min(_pp_var_g[tid_]+weight+_pp_var_weight_plastic[tid_], _pp_var_max_weight[tid_]));
                _pp_var_apre[tid_] = _pp_var_apre[tid_]+_pp_var_Apre[tid_];
                _pp_var_weight_plastic[tid_] = _pp_var_weight_plastic[tid_]+_pp_var_apost[tid_];
            }
        }
    }
}
__global__
void post_event(arb_mechanism_ppack params_) {
    PPACK_IFACE_BLOCK;
    auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid_<_pp_var_width) {
        auto node_index_i_ = _pp_var_node_index[tid_];
        auto cid_ = _pp_var_vec_ci[node_index_i_];
        auto offset_ = _pp_var_n_detectors * cid_;
        for (unsigned c = 0; c < _pp_var_n_detectors; c++) {
            auto time = _pp_var_time_since_spike[offset_ + c];
            if (time >= 0) {
                _pp_var_apost[tid_] = _pp_var_apost[tid_]+_pp_var_Apost[tid_];
                _pp_var_weight_plastic[tid_] = _pp_var_weight_plastic[tid_]+_pp_var_apre[tid_];
            }
        }
    }
}
} // namespace

void mechanism_expsyn_stdp_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 4}, block_dim>>>(*p);
}

void mechanism_expsyn_stdp_gpu_compute_currents_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    compute_currents<<<grid_dim, block_dim>>>(*p);
}

void mechanism_expsyn_stdp_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_expsyn_stdp_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_expsyn_stdp_gpu_post_event_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    post_event<<<grid_dim, block_dim>>>(*p);
}

void mechanism_expsyn_stdp_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* stream_ptr) {
    auto n = stream_ptr->n_streams;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    apply_events<<<grid_dim, block_dim>>>(*p, *stream_ptr);
}

} // namespace default_catalogue
} // namespace arb
