#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace allen_catalogue {
namespace kernel_Kv2like {

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
[[maybe_unused]] auto* _pp_var_h1 = pp->state_vars[1];\
[[maybe_unused]] auto* _pp_var_h2 = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_gbar = pp->parameters[0];\
[[maybe_unused]] auto& _pp_var_ion_k = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_k_index = pp->ion_states[0].index;\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type mBeta, mAlpha, hInf, ll0_, ll1_;
        ll1_ =  0.;
        ll0_ =  0.;
        ll0_ =  43.0-v;
        ll1_ =  11.0*exprelr(ll0_* 0.090909090909090912);
        mAlpha =  0.12*ll1_;
        mBeta =  0.02*exp( -(v+ 1.27)* 0.0083333333333333332);
        hInf =  1.0/( 1.0+exp((v+ 58.0)* 0.090909090909090912));
        _pp_var_m[i_] = mAlpha/(mAlpha+mBeta);
        _pp_var_h1[i_] = hInf;
        _pp_var_h2[i_] = hInf;
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
        _pp_var_m[i_] =  -ba_0_+(_pp_var_m[i_]+ba_0_)*ll3_;
        a_1_ =  -1.0*h1Rat;
        ba_1_ = hInf*h1Rat/a_1_;
        ll4_ = a_1_*dt;
        ll5_ = ( 1.0+ 0.5*ll4_)/( 1.0- 0.5*ll4_);
        _pp_var_h1[i_] =  -ba_1_+(_pp_var_h1[i_]+ba_1_)*ll5_;
        a_2_ =  -1.0*h2Rat;
        ba_2_ = hInf*h2Rat/a_2_;
        ll6_ = a_2_*dt;
        ll7_ = ( 1.0+ 0.5*ll6_)/( 1.0- 0.5*ll6_);
        _pp_var_h2[i_] =  -ba_2_+(_pp_var_h2[i_]+ba_2_)*ll7_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_k_indexi_ = _pp_var_ion_k_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type ek = _pp_var_ion_k.reversal_potential[ion_k_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ik = 0;
        ik =  0.5*_pp_var_gbar[i_]*_pp_var_m[i_]*_pp_var_m[i_]*(_pp_var_h1[i_]+_pp_var_h2[i_])*(v-ek);
        current_ = ik;
        conductivity_ =  0.5*_pp_var_gbar[i_]*_pp_var_m[i_]*_pp_var_m[i_]*(_pp_var_h1[i_]+_pp_var_h2[i_]);
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[i_], conductivity_, _pp_var_vec_g[node_indexi_]);
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
} // namespace kernel_Kv2like
} // namespace allen_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_allen_catalogue_Kv2like_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::allen_catalogue::kernel_Kv2like::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::allen_catalogue::kernel_Kv2like::min_align_;
    result.init_mechanism = arb::allen_catalogue::kernel_Kv2like::init;
    result.compute_currents = arb::allen_catalogue::kernel_Kv2like::compute_currents;
    result.apply_events = arb::allen_catalogue::kernel_Kv2like::apply_events;
    result.advance_state = arb::allen_catalogue::kernel_Kv2like::advance_state;
    result.write_ions = arb::allen_catalogue::kernel_Kv2like::write_ions;
    result.post_event = arb::allen_catalogue::kernel_Kv2like::post_event;
    return &result;
  }}

