#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace bbp_catalogue {
namespace kernel_SK_E2 {

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
[[maybe_unused]] auto _pp_var_zTau = pp->globals[0];\
[[maybe_unused]] auto* _pp_var_z = pp->state_vars[0];\
[[maybe_unused]] auto* _pp_var_gSK_E2bar = pp->parameters[0];\
[[maybe_unused]] auto& _pp_var_ion_k = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_k_index = pp->ion_states[0].index;\
[[maybe_unused]] auto& _pp_var_ion_ca = pp->ion_states[1];\
[[maybe_unused]] auto* _pp_var_ion_ca_index = pp->ion_states[1].index;\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[i_];
        arb_value_type cai = _pp_var_ion_ca.internal_concentration[ion_ca_indexi_];
        if (cai< 9.9999999999999995e-08) {
            _pp_var_z[i_] =  0.;
        }
        else {
            _pp_var_z[i_] =  1.0/( 1.0+pow( 0.00042999999999999999/cai,  4.7999999999999998));
        }
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 1; ++ix) {
        for (arb_size_type iy = 0; iy < _pp_var_width; ++iy) {
            pp->state_vars[ix][iy] *= _pp_var_multiplicity[iy];
        }
    }
}

static void advance_state(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_ca_indexi_ = _pp_var_ion_ca_index[i_];
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type dt = _pp_var_vec_dt[node_indexi_];
        arb_value_type cai = _pp_var_ion_ca.internal_concentration[ion_ca_indexi_];
        arb_value_type a_0_, ll0_, ll2_, ll1_, b_0_;
        ll2_ =  0.;
        ll1_ =  0.;
        ll0_ =  0.;
        if (cai< 9.9999999999999995e-08) {
            ll0_ =  0.;
        }
        else {
            ll0_ =  1.0/( 1.0+pow( 0.00042999999999999999/cai,  4.7999999999999998));
        }
        a_0_ = _pp_var_zTau;
        b_0_ = ll0_;
        ll1_ =  -dt/a_0_;
        ll2_ = ( 1.0+ 0.5*ll1_)/( 1.0- 0.5*ll1_);
        _pp_var_z[i_] = b_0_+(_pp_var_z[i_]-b_0_)*ll2_;
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
        ik = _pp_var_gSK_E2bar[i_]*_pp_var_z[i_]*(v-ek);
        current_ = ik;
        conductivity_ = _pp_var_gSK_E2bar[i_]*_pp_var_z[i_];
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
} // namespace kernel_SK_E2
} // namespace bbp_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_bbp_catalogue_SK_E2_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::bbp_catalogue::kernel_SK_E2::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::bbp_catalogue::kernel_SK_E2::min_align_;
    result.init_mechanism = arb::bbp_catalogue::kernel_SK_E2::init;
    result.compute_currents = arb::bbp_catalogue::kernel_SK_E2::compute_currents;
    result.apply_events = arb::bbp_catalogue::kernel_SK_E2::apply_events;
    result.advance_state = arb::bbp_catalogue::kernel_SK_E2::advance_state;
    result.write_ions = arb::bbp_catalogue::kernel_SK_E2::write_ions;
    result.post_event = arb::bbp_catalogue::kernel_SK_E2::post_event;
    return &result;
  }}

