#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace default_catalogue {
namespace kernel_hh {

using ::arb::math::exprelr;
using ::arb::math::safeinv;
using ::std::abs;
using ::std::cos;
using ::std::exp;
using ::std::log;
using ::std::max;
using ::std::min;
using ::std::pow;
using ::std::sin;

static constexpr unsigned simd_width_ = 1;
static constexpr unsigned min_align_ = std::max(alignof(arb_value_type), alignof(arb_index_type));

#define PPACK_IFACE_BLOCK \
[[maybe_unused]] auto  _pp_var_width             = pp->width;\
[[maybe_unused]] auto  _pp_var_n_detectors       = pp->n_detectors;\
[[maybe_unused]] auto* _pp_var_vec_ci            = pp->vec_ci;\
[[maybe_unused]] auto* _pp_var_vec_di            = pp->vec_di;\
[[maybe_unused]] auto* _pp_var_vec_t             = pp->vec_t;\
[[maybe_unused]] auto* _pp_var_vec_dt            = pp->vec_dt;\
[[maybe_unused]] auto* _pp_var_vec_v             = pp->vec_v;\
[[maybe_unused]] auto* _pp_var_vec_i             = pp->vec_i;\
[[maybe_unused]] auto* _pp_var_vec_g             = pp->vec_g;\
[[maybe_unused]] auto* _pp_var_temperature_degC  = pp->temperature_degC;\
[[maybe_unused]] auto* _pp_var_diam_um           = pp->diam_um;\
[[maybe_unused]] auto* _pp_var_time_since_spike  = pp->time_since_spike;\
[[maybe_unused]] auto* _pp_var_node_index        = pp->node_index;\
[[maybe_unused]] auto* _pp_var_peer_index        = pp->peer_index;\
[[maybe_unused]] auto* _pp_var_multiplicity      = pp->multiplicity;\
[[maybe_unused]] auto* _pp_var_weight            = pp->weight;\
[[maybe_unused]] auto& _pp_var_events            = pp->events;\
[[maybe_unused]] auto& _pp_var_mechanism_id      = pp->mechanism_id;\
[[maybe_unused]] auto& _pp_var_index_constraints = pp->index_constraints;\
[[maybe_unused]] auto* _pp_var_m = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_h = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_n = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_q10 = pp->state_vars[3];\
[[maybe_unused]] auto* _pp_var_gnabar = pp->parameters[0];\
[[maybe_unused]] auto* _pp_var_gkbar = pp->parameters[1];\
[[maybe_unused]] auto* _pp_var_gl = pp->parameters[2];\
[[maybe_unused]] auto* _pp_var_el = pp->parameters[3];\
[[maybe_unused]] auto& _pp_var_ion_na = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_na_index = pp->ion_states[0].index;\
[[maybe_unused]] auto& _pp_var_ion_k = pp->ion_states[1];\
[[maybe_unused]] auto* _pp_var_ion_k_index = pp->ion_states[1].index;\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type celsius = _pp_var_temperature_degC[node_indexi_];
        arb_value_type r_3_, r_2_, r_1_, r_0_, beta, alpha;
        _pp_var_q10[i_] = pow( 3.0, (celsius- 6.2999999999999998)* 0.10000000000000001);
        r_0_ =  0.;
        r_1_ =  0.;
        r_1_ =  -(v+ 40.0);
        r_0_ =  10.0*exprelr(r_1_* 0.10000000000000001);
        alpha =  0.10000000000000001*r_0_;
        beta =  4.0*exp( -(v+ 65.0)* 0.055555555555555552);
        _pp_var_m[i_] = alpha/(alpha+beta);
        alpha =  0.070000000000000007*exp( -(v+ 65.0)* 0.050000000000000003);
        beta =  1.0/(exp( -(v+ 35.0)* 0.10000000000000001)+ 1.0);
        _pp_var_h[i_] = alpha/(alpha+beta);
        r_2_ =  0.;
        r_3_ =  0.;
        r_3_ =  -(v+ 55.0);
        r_2_ =  10.0*exprelr(r_3_* 0.10000000000000001);
        alpha =  0.01*r_2_;
        beta =  0.125*exp( -(v+ 65.0)* 0.012500000000000001);
        _pp_var_n[i_] = alpha/(alpha+beta);
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 3; ++ix) {
        for (arb_size_type iy = 0; iy < _pp_var_width; ++iy) {
            pp->state_vars[ix][iy] *= _pp_var_multiplicity[iy];
        }
    }
}

static void advance_state(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type a_2_, r_2_, ba_0_, a_0_, alpha, ll5_, ll0_, beta, ba_1_, ll3_, ll1_, ba_2_, r_3_, ll2_, r_0_, r_1_, sum, a_1_, ll4_;
        ll5_ =  0.;
        ll4_ =  0.;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        r_0_ =  0.;
        r_1_ =  0.;
        r_1_ =  -(v+ 40.0);
        r_0_ =  10.0*exprelr(r_1_* 0.10000000000000001);
        alpha =  0.10000000000000001*r_0_;
        beta =  4.0*exp( -(v+ 65.0)* 0.055555555555555552);
        sum = alpha+beta;
        a_0_ =  -sum*_pp_var_q10[i_];
        ba_0_ = alpha*_pp_var_q10[i_]/a_0_;
        ll0_ = a_0_*dt;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_m[i_] =  -ba_0_+(_pp_var_m[i_]+ba_0_)*ll1_;
        alpha =  0.070000000000000007*exp( -(v+ 65.0)* 0.050000000000000003);
        beta =  1.0/(exp( -(v+ 35.0)* 0.10000000000000001)+ 1.0);
        sum = alpha+beta;
        a_1_ =  -sum*_pp_var_q10[i_];
        ba_1_ = alpha*_pp_var_q10[i_]/a_1_;
        ll2_ = a_1_*dt;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_h[i_] =  -ba_1_+(_pp_var_h[i_]+ba_1_)*ll3_;
        r_2_ =  0.;
        r_3_ =  0.;
        r_3_ =  -(v+ 55.0);
        r_2_ =  10.0*exprelr(r_3_* 0.10000000000000001);
        alpha =  0.01*r_2_;
        beta =  0.125*exp( -(v+ 65.0)* 0.012500000000000001);
        sum = alpha+beta;
        a_2_ =  -sum*_pp_var_q10[i_];
        ba_2_ = alpha*_pp_var_q10[i_]/a_2_;
        ll4_ = a_2_*dt;
        ll5_ = ( 1.0+ 0.5*ll4_)/( 1.0- 0.5*ll4_);
        _pp_var_n[i_] =  -ba_2_+(_pp_var_n[i_]+ba_2_)*ll5_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_k_indexi_ = _pp_var_ion_k_index[i_];
        auto ion_na_indexi_ = _pp_var_ion_na_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type conductivity_ = 0;
        arb_value_type ina = 0;
        arb_value_type current_ = 0;
        arb_value_type ek = _pp_var_ion_k.reversal_potential[ion_k_indexi_];
        arb_value_type ik = 0;
        arb_value_type ena = _pp_var_ion_na.reversal_potential[ion_na_indexi_];
        arb_value_type il = 0;
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type gk, n2, m_, n_;
        n_ = _pp_var_n[i_];
        m_ = _pp_var_m[i_];
        n2 = n_*n_;
        gk = _pp_var_gkbar[i_]*n2*n2;
        ina = _pp_var_gnabar[i_]*m_*m_*m_*_pp_var_h[i_]*(v-ena);
        ik = gk*(v-ek);
        il = _pp_var_gl[i_]*(v-_pp_var_el[i_]);
        current_ = ik+il+ina;
        conductivity_ = _pp_var_gnabar[i_]*m_*m_*m_*_pp_var_h[i_]+gk+_pp_var_gl[i_];
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[i_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_ion_na.current_density[ion_na_indexi_] = fma(10.0*_pp_var_weight[i_], ina, _pp_var_ion_na.current_density[ion_na_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[i_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_k.current_density[ion_k_indexi_] = fma(10.0*_pp_var_weight[i_], ik, _pp_var_ion_k.current_density[ion_k_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_hh
} // namespace default_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_default_catalogue_hh_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::default_catalogue::kernel_hh::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::default_catalogue::kernel_hh::min_align_;
    result.init_mechanism = arb::default_catalogue::kernel_hh::init;
    result.compute_currents = arb::default_catalogue::kernel_hh::compute_currents;
    result.apply_events = arb::default_catalogue::kernel_hh::apply_events;
    result.advance_state = arb::default_catalogue::kernel_hh::advance_state;
    result.write_ions = arb::default_catalogue::kernel_hh::write_ions;
    result.post_event = arb::default_catalogue::kernel_hh::post_event;
    return &result;
  }}

