#include <arbor/gpu/gpu_common.hpp>
#include <arbor/gpu/math_cu.hpp>
#include <arbor/gpu/reduce_by_key.hpp>
#include <arbor/mechanism_abi.h>

namespace testing {

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
auto _pp_var_gbar __attribute__((unused)) = params_.globals[0];\
auto _pp_var_gl __attribute__((unused)) = params_.globals[1];\
auto _pp_var_eca __attribute__((unused)) = params_.globals[2];\
auto _pp_var_el __attribute__((unused)) = params_.globals[3];\
auto* _pp_var_m __attribute__((unused)) = params_.state_vars[0];\
auto* _pp_var_h __attribute__((unused)) = params_.state_vars[1];\
auto* _pp_var_s __attribute__((unused)) = params_.state_vars[2];\
auto* _pp_var_d __attribute__((unused)) = params_.state_vars[3];\
auto& _pp_var_ion_ca __attribute__((unused)) = params_.ion_states[0];\
auto* _pp_var_ion_ca_index __attribute__((unused)) = params_.ion_states[0].index;\
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
        arb_value_type vrest, k;
        vrest =  -65.0;
        k = pow( 0.25+exp((v+ 83.5)* 0.15873015873015872),  0.5)- 0.5;
        _pp_var_m[tid_] =  1.0/( 1.0+exp( -(vrest+ 63.0)* 0.12820512820512822));
        _pp_var_h[tid_] =  1.0/( 1.0+k+pow(k,  2.0));
        _pp_var_d[tid_] = _pp_var_h[tid_]*pow(k,  2.0);
        _pp_var_s[tid_] =  1.0-_pp_var_h[tid_]-_pp_var_d[tid_];
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
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type t_7_, t_3_, t_2_, t_1_, dsh_q10, a_6_, t_0_, a_5_, t_5_, a_4_, a_0_, a_3_, t_4_, a_1_, a_2_, a_7_, alpha1, k, beta2, beta1, mi, taum, t_8_, m_q10, ba_0_, t_6_, ll0_, alpha2, ll1_;
        ll1_ =  0.;
        ll0_ =  0.;
        m_q10 =  5.0;
        mi =  1.0/( 1.0+exp( -(v+ 63.0)* 0.12820512820512822));
        taum = ( 1.7+exp( -(v+ 28.800000000000001)* 0.07407407407407407))*mi;
        a_0_ = m_q10* -1.0/taum;
        ba_0_ = m_q10*mi/taum/a_0_;
        ll0_ = a_0_*dt;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_m[tid_] =  -ba_0_+(_pp_var_m[tid_]+ba_0_)*ll1_;
        dsh_q10 =  3.0;
        k = pow( 0.25+exp((v+ 83.5)* 0.15873015873015872),  0.5)- 0.5;
        alpha1 = dsh_q10*exp( -(v+ 160.30000000000001)* 0.056179775280898875);
        beta1 = alpha1*k;
        alpha2 = dsh_q10*( 1.0+exp((v+ 37.399999999999999)* 0.033333333333333333))* 0.0041666666666666666/( 1.0+k);
        beta2 = alpha2*k;
        a_1_ =  1.0- -1.0*alpha2*dt;
        a_2_ =  -( -1.0* -beta2*dt);
        a_3_ =  1.0- -beta1*dt;
        a_4_ =  -(alpha1*dt);
        a_5_ =  -(alpha2*dt);
        a_6_ =  -( -1.0* -beta1*dt);
        a_7_ =  1.0-( -1.0*alpha1+ -beta2)*dt;
        t_0_ = a_3_*a_5_;
        t_1_ = a_3_*a_7_-a_6_*a_4_;
        t_2_ = a_3_*_pp_var_s[tid_]-a_6_*_pp_var_h[tid_];
        t_3_ = a_1_*t_1_-t_0_*a_2_;
        t_4_ = a_1_*t_2_-t_0_*_pp_var_d[tid_];
        t_5_ = t_3_*a_1_;
        t_6_ = t_3_*_pp_var_d[tid_]-a_2_*t_4_;
        t_7_ = t_3_*a_3_;
        t_8_ = t_3_*_pp_var_h[tid_]-a_4_*t_4_;
        _pp_var_d[tid_] = t_6_/t_5_;
        _pp_var_h[tid_] = t_8_/t_7_;
        _pp_var_s[tid_] = t_4_/t_3_;
    }
}

__global__
void compute_currents(arb_mechanism_ppack params_) {
    int n_ = params_.width;
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    PPACK_IFACE_BLOCK;
    if (tid_<n_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[tid_];
        auto node_indexi_ = _pp_var_node_index[tid_];
        arb_value_type current_ = 0;
        arb_value_type conductivity_ = 0;
        arb_value_type il = 0;
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ica = 0;
        ica = _pp_var_gbar*pow(_pp_var_m[tid_],  3.0)*_pp_var_h[tid_]*(v-_pp_var_eca);
        il = _pp_var_gl*(v-_pp_var_el);
        current_ = ica+il;
        conductivity_ = _pp_var_gl+_pp_var_gbar*pow(_pp_var_m[tid_],  3.0)*_pp_var_h[tid_];
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[tid_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[tid_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_ion_ca.current_density[ion_ca_indexi_] = fma(10.0*_pp_var_weight[tid_], ica, _pp_var_ion_ca.current_density[ion_ca_indexi_]);
    }
}

} // namespace

void mechanism_test_kinlva_gpu_init_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    init<<<grid_dim, block_dim>>>(*p);
    if (!p->multiplicity) return;
    multiply<<<dim3{grid_dim, 4}, block_dim>>>(*p);
}

void mechanism_test_kinlva_gpu_compute_currents_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    compute_currents<<<grid_dim, block_dim>>>(*p);
}

void mechanism_test_kinlva_gpu_advance_state_(arb_mechanism_ppack* p) {
    auto n = p->width;
    unsigned block_dim = 128;
    unsigned grid_dim = ::arb::gpu::impl::block_count(n, block_dim);
    advance_state<<<grid_dim, block_dim>>>(*p);
}

void mechanism_test_kinlva_gpu_write_ions_(arb_mechanism_ppack* p) {}

void mechanism_test_kinlva_gpu_post_event_(arb_mechanism_ppack* p) {}
void mechanism_test_kinlva_gpu_apply_events_(arb_mechanism_ppack* p, arb_deliverable_event_stream* events) {}

} // namespace testing
