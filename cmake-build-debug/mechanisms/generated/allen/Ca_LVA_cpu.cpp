#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace allen_catalogue {
namespace kernel_Ca_LVA {

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
[[maybe_unused]] auto* _pp_var_qt = pp->state_vars[2];\
[[maybe_unused]] auto* _pp_var_gbar = pp->parameters[0];\
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
        arb_value_type celsius = _pp_var_temperature_degC[node_indexi_];
        _pp_var_qt[i_] = pow( 2.2999999999999998, (celsius- 21.0)* 0.10000000000000001);
        _pp_var_m[i_] =  1.0/( 1.0+exp((v+ 40.0)* -0.16666666666666666));
        _pp_var_h[i_] =  1.0/( 1.0+exp((v+ 90.0)* 0.15625));
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 2; ++ix) {
        for (arb_size_type iy = 0; iy < _pp_var_width; ++iy) {
            pp->state_vars[ix][iy] *= _pp_var_multiplicity[iy];
        }
    }
}

static void advance_state(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type a_1_, a_0_, hRat, mRat, mInf, hInf, ba_0_, ll0_, ba_1_, ll3_, ll1_, ll2_;
        ll3_ =  0.;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        mInf =  1.0/( 1.0+exp((v+ 40.0)* -0.16666666666666666));
        hInf =  1.0/( 1.0+exp((v+ 90.0)* 0.15625));
        mRat = _pp_var_qt[i_]/( 5.0+ 20.0/( 1.0+exp((v+ 35.0)* 0.20000000000000001)));
        hRat = _pp_var_qt[i_]/( 20.0+ 50.0/( 1.0+exp((v+ 50.0)* 0.14285714285714285)));
        a_0_ =  -1.0*mRat;
        ba_0_ = mInf*mRat/a_0_;
        ll0_ = a_0_*dt;
        ll1_ = ( 1.0+ 0.5*ll0_)/( 1.0- 0.5*ll0_);
        _pp_var_m[i_] =  -ba_0_+(_pp_var_m[i_]+ba_0_)*ll1_;
        a_1_ =  -1.0*hRat;
        ba_1_ = hInf*hRat/a_1_;
        ll2_ = a_1_*dt;
        ll3_ = ( 1.0+ 0.5*ll2_)/( 1.0- 0.5*ll2_);
        _pp_var_h[i_] =  -ba_1_+(_pp_var_h[i_]+ba_1_)*ll3_;
    }
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type conductivity_ = 0;
        arb_value_type current_ = 0;
        arb_value_type eca = _pp_var_ion_ca.reversal_potential[ion_ca_indexi_];
        arb_value_type v = _pp_var_vec_v[node_indexi_];
        arb_value_type ica = 0;
        ica = _pp_var_gbar[i_]*_pp_var_m[i_]*_pp_var_m[i_]*_pp_var_h[i_]*(v-eca);
        current_ = ica;
        conductivity_ = _pp_var_gbar[i_]*_pp_var_m[i_]*_pp_var_m[i_]*_pp_var_h[i_];
        _pp_var_vec_g[node_indexi_] = fma(10.0*_pp_var_weight[i_], conductivity_, _pp_var_vec_g[node_indexi_]);
        _pp_var_vec_i[node_indexi_] = fma(10.0*_pp_var_weight[i_], current_, _pp_var_vec_i[node_indexi_]);
        _pp_var_ion_ca.current_density[ion_ca_indexi_] = fma(10.0*_pp_var_weight[i_], ica, _pp_var_ion_ca.current_density[ion_ca_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_Ca_LVA
} // namespace allen_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_allen_catalogue_Ca_LVA_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::allen_catalogue::kernel_Ca_LVA::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::allen_catalogue::kernel_Ca_LVA::min_align_;
    result.init_mechanism = arb::allen_catalogue::kernel_Ca_LVA::init;
    result.compute_currents = arb::allen_catalogue::kernel_Ca_LVA::compute_currents;
    result.apply_events = arb::allen_catalogue::kernel_Ca_LVA::apply_events;
    result.advance_state = arb::allen_catalogue::kernel_Ca_LVA::advance_state;
    result.write_ions = arb::allen_catalogue::kernel_Ca_LVA::write_ions;
    result.post_event = arb::allen_catalogue::kernel_Ca_LVA::post_event;
    return &result;
  }}

