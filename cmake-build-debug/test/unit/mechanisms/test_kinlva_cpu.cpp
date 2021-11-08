#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace testing {
namespace kernel_test_kinlva {

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
[[maybe_unused]] auto _pp_var_gbar = pp->globals[0];\
[[maybe_unused]] auto _pp_var_gl = pp->globals[1];\
[[maybe_unused]] auto _pp_var_eca = pp->globals[2];\
[[maybe_unused]] auto _pp_var_el = pp->globals[3];\
[[maybe_unused]] auto* _pp_var_m = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_h = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_s = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_d = pp->state_vars[3];\
[[maybe_unused]] auto& _pp_var_ion_ca = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_ca_index = pp->ion_states[0].index;\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type vrest, k;
        vrest =  -65.0;
        k = pow( 0.25+exp((v+ 83.5)* 0.15873015873015872),  0.5)- 0.5;
        _pp_var_m[i_] =  1.0/( 1.0+exp( -(vrest+ 63.0)* 0.12820512820512822));
        _pp_var_h[i_] =  1.0/( 1.0+k+pow(k,  2.0));
        _pp_var_d[i_] = _pp_var_h[i_]*pow(k,  2.0);
        _pp_var_s[i_] =  1.0-_pp_var_h[i_]-_pp_var_d[i_];
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 4; ++ix) {
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
        _pp_var_m[i_] =  -ba_0_+(_pp_var_m[i_]+ba_0_)*ll1_;
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
        t_2_ = a_3_*_pp_var_s[i_]-a_6_*_pp_var_h[i_];
        t_3_ = a_1_*t_1_-t_0_*a_2_;
        t_4_ = a_1_*t_2_-t_0_*_pp_var_d[i_];
        t_5_ = t_3_*a_1_;
        t_6_ = t_3_*_pp_var_d[i_]-a_2_*t_4_;
        t_7_ = t_3_*a_3_;
        t_8_ = t_3_*_pp_var_h[i_]-a_4_*t_4_;
        _pp_var_d[i_] = t_6_/t_5_;
        _pp_var_h[i_] = t_8_/t_7_;
        _pp_var_s[i_] = t_4_/t_3_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type current_ = 0;
        arb_value_type conductivity_ = 0;
        arb_value_type il = 0;
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ica = 0;
        ica = _pp_var_gbar*pow(_pp_var_m[i_],  3.0)*_pp_var_h[i_]*(v-_pp_var_eca);
        il = _pp_var_gl*(v-_pp_var_el);
        current_ = ica+il;
        conductivity_ = _pp_var_gl+_pp_var_gbar*pow(_pp_var_m[i_],  3.0)*_pp_var_h[i_];
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[i_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[i_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_ion_ca.current_density[ion_ca_indexi_] = fma(10.0*_pp_var_weight[i_], ica, _pp_var_ion_ca.current_density[ion_ca_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_test_kinlva
} // namespace testing

extern "C" {
  arb_mechanism_interface* make_testing_test_kinlva_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = testing::kernel_test_kinlva::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = testing::kernel_test_kinlva::min_align_;
    result.init_mechanism = testing::kernel_test_kinlva::init;
    result.compute_currents = testing::kernel_test_kinlva::compute_currents;
    result.apply_events = testing::kernel_test_kinlva::apply_events;
    result.advance_state = testing::kernel_test_kinlva::advance_state;
    result.write_ions = testing::kernel_test_kinlva::write_ions;
    result.post_event = testing::kernel_test_kinlva::post_event;
    return &result;
  }}

