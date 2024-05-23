#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>

#include "Stream.h"

#include <core/Device.h>
#include <core/Stream.h>
#include <structmember.h>

namespace torch_ipex::xpu {

PyObject* THDPStreamClass = nullptr;

static PyObject* THDPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const auto current_device = torch_ipex::xpu::dpcpp::current_device();

  int priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  constexpr const char* kwlist[] = {
      "priority", "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|iLLL",
          const_cast<char**>(kwlist),
          &priority,
          &stream_id,
          &device_index,
          &device_type)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  torch_ipex::xpu::dpcpp::DPCPPStream stream = (stream_id || device_index || device_type)
      ? torch_ipex::xpu::dpcpp::DPCPPStream::unpack3(
            stream_id, device_index, static_cast<c10::DeviceType>(device_type))
      : torch_ipex::xpu::dpcpp::getStreamFromPool(
            /* isHighPriority */ priority < 0 ? true : false, current_device);

  THDPStream* self = (THDPStream*)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  new (&self->dpcpp_stream) torch_ipex::xpu::dpcpp::DPCPPStream(stream);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THDPStream_dealloc(THDPStream* self) {
  self->dpcpp_stream.~DPCPPStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THDPStream_get_device(THDPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->dpcpp_stream.device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPStream_get_sycl_queue(THDPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  // NOTE: Here is a high dependency on the implementation of queue pool using
  // smart pointer in runtime.
  return PyCapsule_New(
      self->dpcpp_stream.queue(), "torch.xpu.Stream.sycl_queue", nullptr);
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPStream_get_priority(THDPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  //  return PyLong_FromLong(self->dpcpp_stream.priority());
  return PyLong_FromLong(0);
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPStream_priority_range() {
  HANDLE_TH_ERRORS
  int least_priority = 0, greatest_priority = 0;
  //  std::tie(least_priority, greatest_priority) =
  //  xpu::XPUStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPStream_query(THDPStream* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  //  return PyBool_FromLong(self->dpcpp_stream.query());
  return PyBool_FromLong(0);
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPStream_synchronize(THDPStream* self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    self->dpcpp_stream.synchronize_and_throw();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THDPStream_eq(THDPStream* self, THDPStream* other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->dpcpp_stream == other->dpcpp_stream);
  END_HANDLE_TH_ERRORS
}

static struct PyMemberDef THDPStream_members[] = {{nullptr}};

static struct PyGetSetDef THDPStream_properties[] = {
    {"sycl_queue",
     (getter)THDPStream_get_sycl_queue,
     nullptr,
     nullptr,
     nullptr},
    {"priority", (getter)THDPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}};

static PyMethodDef THDPStream_methods[] = {
    {(char*)"query", (PyCFunction)THDPStream_query, METH_NOARGS, nullptr},
    {(char*)"synchronize",
     (PyCFunction)THDPStream_synchronize,
     METH_NOARGS,
     nullptr},
    {(char*)"priority_range",
     (PyCFunction)(void (*)(void))THDPStream_priority_range,
     METH_STATIC | METH_NOARGS,
     nullptr},
    {(char*)"__eq__", (PyCFunction)THDPStream_eq, METH_O, nullptr},
    {nullptr}};

PyTypeObject THDPStreamType = {
    PyVarObject_HEAD_INIT(
        &PyType_Type,
        0) "torch._C._XPUStreamBase", /* tp_name */
    sizeof(THDPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THDPStream_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    0, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash  */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    THDPStream_methods, /* tp_methods */
    THDPStream_members, /* tp_members */
    THDPStream_properties, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    0, /* tp_init */
    0, /* tp_alloc */
    THDPStream_pynew, /* tp_new */
};

void THDPStream_init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THDPStreamType.tp_base = THPStreamClass;
  THDPStreamClass = (PyObject*)&THDPStreamType;
  if (PyType_Ready(&THDPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THDPStreamType);
  if (PyModule_AddObject(module, "_XPUStreamBase", (PyObject*)&THDPStreamType) <
      0) {
    throw python_error();
  }
}

} // namespace torch_ipex::xpu
