#include <arbor/gpu/gpu_common.hpp>
#include <arbor/gpu/math_cu.hpp>
#include <arbor/gpu/reduce_by_key.hpp>
#include <arbor/mechanism_abi.h>

namespace arb {
namespace allen_catalogue {

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
auto* _pp_var_m __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_h1 __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_h2 __attribute__((unused)) = params_.state_vars[2];\
auto* _pp_var_gbar __attribute__((unused)) = params_.parameters[0];\
auto& _pp_var_ion_k __attribute__((unused)) = params_.ion_states[0];\
auto* _pp_var_ion_k_index __attribute__((unused)) = params_.ion_states[0].index;\
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
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type mBeta, mAlpha, hInf, ll0_, ll1_;
        ll1_ =  0.;
        ll0_ =  0.;
        ll0_ =  43.0-v;
        ll1_ =  11.0*exprelr(ll0_* 0.090909090909090912);
        mAlpha =  0.12*ll1_;
        mBeta =  0.02*exp( -(v+ 1.27)* 0.0083333333333333332);
        hInf =  1.0/( 1.0+exp((v+ 58.0)* 0.090909090909090912));
        _pp_var_m[tid_] = mAlpha/(mAlpha+mBeta);
        _pp_var_h1[tid_] = hInf;
        _pp_var_h2[tid_] = hInf;
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
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type celsius = _pp_var_temperature_degC[node_indexi_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type a_1_, a_2_, a_0_, ll4_, qt, mRat, mBeta, hInf, ba_0_, ba_1_, ll3_, ll6_, h1Rat, ll1_, ba_2_, ll2_, mAlpha, h2Rat, ll5_, ll0_, ll7_;
        ll7_ =  0.;
        ll6_ =  0.;
        ll5_ =  0.;
        ll4_ =  0.;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        qt = pow( 2.2999999999999998, (celsius- 21.0)* 0.10000000000000001);
        ll0_ =  43.0-v;
        ll1_ =  11.0*exprelr(ll0_* 0.090909090909090912);
        mAlpha =  0.12*ll1_;
        mBeta =  0.02*exp( -(v+ 1.27)* 0.0083333333333333332);
        mRat =  0.40000000000000002*qt*(mAlpha+mBeta);
        hInf =  1.0/( 1.0+exp((v+ 58.0)* 0.090909090909090912));
        h1Rat = qt/( 360.0+( 1010.0+ 23.699999999999999*(v+ 54.0))*exp(pow( -((v+ 75.0)* 0.020833333333333332),  2.0)));
        h2Rat = qt/( 2350.0+ 1380.0*exp( -0.010999999999999999*v)- 210.0*exp( -0.029999999999999999*v));
        if (h2Rat< 0.) {
            h2Rat =  0.001;
        }
        a_0_ =  -mRat;
        ba_0_ =  0.40000000000000002*qt*mAlpha/a_0_;
        ll2_ = a_0_*dt;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_m[tid_] =  -ba_0_+(_pp_var_m[tid_]+ba_0_)*ll3_;
        a_1_ =  -1.0*h1Rat;
        ba_1_ = hInf*h1Rat/a_1_;
        ll4_ = a_1_*dt;
        ll5_ = ( 1.0+ 0.5*ll4_)/( 1.0- 0.5*ll4_);
        _pp_var_h1[tid_] =  -ba_1_+(_pp_var_h1[tid_]+ba_1_)*ll5_;
        a_2_ =  -1.0*h2Rat;
        ba_2_ = hInf*h2Rat/a_2_;
        ll6_ = a_2_*dt;
        ll7_ = ( 1.0+ 0.5*ll6_)/( 1.0- 0.5*ll6_);
        _pp_var_h2[tid_] =  -ba_2_+(_pp_var_h2[tid_]+ba_2_)*ll7_;
    }
}

__global__
void compute_currents(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto ion_k_indexi_ = _pp_var_ion_k_index[tid_];
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type ek = _pp_var_ion_k.reversal_potential[ion_k_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ik = 0;
        ik =  0.5*_pp_var_gbar[tid_]*_pp_var_m[tid_]*_pp_var_m[tid_]*(_pp_var_h1[tid_]+_pp_var_h2[tid_])*(v-ek);
        current_ = ik;
        conductivity_ =  0.5*_pp_var_gbar[tid_]*_pp_var_m[tid_]*_pp_var_m[tid_]*(_pp_var_h1[tid_]+_pp_var_h2[tid_]);
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[tid_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[tid_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_k.current_density[ion_k_indexi_] = fma(10.0*_pp_var_weight[tid_], ik, _pp_var_ion_k.current_density[ion_k_indexi_]);
    }
}

} // namespace

void mechanism_Kv2like_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 3}, block_dim>>>(*p);
}

void mechanism_Kv2like_gpu_compute_currents_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    compute_currents<<<grid_dim, block_dim>>>(*p);
}

void mechanism_Kv2like_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_Kv2like_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_Kv2like_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_Kv2like_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace allen_catalogue
} // namespace arb
