#pragma once

#include "pybind11/pybind11.h"

namespace xpu {
namespace onednn {

void initOnednnGraphPythonBindings(pybind11::module& m);

} // namespace onednn
} // namespace xpu