#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

namespace arb {
namespace default_catalogue {
namespace kernel_nernst {

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
[[maybe_unused]] auto _pp_var_R = pp->globals[0];\
[[maybe_unused]] auto _pp_var_F = pp->globals[1];\
[[maybe_unused]] auto* _pp_var_coeff = pp->state_vars[0];\
[[maybe_unused]] auto& _pp_var_ion_x = pp->ion_states[0];\
[[maybe_unused]] auto* _pp_var_ion_x_index = pp->ion_states[0].index;\
//End of IFACEBLOCK

// procedure prototypes

// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type zx = _pp_var_ion_x.ionic_charge[0];
        arb_value_type celsius = _pp_var_temperature_degC[node_indexi_];
        _pp_var_coeff[i_] = _pp_var_R*(celsius+ 273.14999999999998)/(zx*_pp_var_F)* 1000.0;
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 0; ++ix) {
        for (arb_size_type iy = 0; iy < _pp_var_width; ++iy) {
            pp->state_vars[ix][iy] *= _pp_var_multiplicity[iy];
        }
    }
}

static void advance_state(arb_mechanism_ppack* pp) {
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto ion_x_indexi_ = _pp_var_ion_x_index[i_];
        arb_value_type xi = _pp_var_ion_x.internal_concentration[ion_x_indexi_];
        arb_value_type xo = _pp_var_ion_x.external_concentration[ion_x_indexi_];
        arb_value_type ex = 0;
        ex = _pp_var_coeff[i_]*log(xo/xi);
        _pp_var_ion_x.reversal_potential[ion_x_indexi_] = ex;
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}

// Procedure definitions
#undef PPACK_IFACE_BLOCK
} // namespace kernel_nernst
} // namespace default_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_default_catalogue_nernst_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::default_catalogue::kernel_nernst::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::default_catalogue::kernel_nernst::min_align_;
    result.init_mechanism = arb::default_catalogue::kernel_nernst::init;
    result.compute_currents = arb::default_catalogue::kernel_nernst::compute_currents;
    result.apply_events = arb::default_catalogue::kernel_nernst::apply_events;
    result.advance_state = arb::default_catalogue::kernel_nernst::advance_state;
    result.write_ions = arb::default_catalogue::kernel_nernst::write_ions;
    result.post_event = arb::default_catalogue::kernel_nernst::post_event;
    return &result;
  }}

