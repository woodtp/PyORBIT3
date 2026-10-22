#include "wrap_linac_tracking.hh"

#include "main/pyORBIT_Object.hh"
#include "methodobject.h"
#include "orbit/wrap_bunch.hh"
#include "tracking/linac_tracking.hh"

namespace wrap_linac_tracking
{

namespace
{
Bunch* parse_bunch(PyObject* obj, const char* err_msg)
{
    if (!PyObject_TypeCheck(obj, wrap_orbit_bunch::bunchType()))
    {
        PyErr_SetString(PyExc_TypeError, err_msg);
        return nullptr;
    }
    auto* bunch = static_cast<Bunch*>(reinterpret_cast<pyORBIT_Object*>(obj)->cpp_obj);
    if (bunch == nullptr)
    {
        PyErr_SetString(PyExc_RuntimeError, "Bunch is not initialized");
    }
    return bunch;
}
} // namespace

#ifdef __cplusplus
extern "C"
{
#endif

    //---------------------------------------------------------
    //linac_tracking method wrappers
    //---------------------------------------------------------

    //Tracking a bunch through a linac drift element wrapper
    static PyObject* wrap_linac_drift(PyObject* Py_UNUSED(self), PyObject* const* args,
                                      Py_ssize_t nargs)
    {
        if (nargs != 2)
        {
            PyErr_Format(PyExc_TypeError,
                         "linac_drift(Bunch, length) takes 2 positional arguments (%zd given).",
                         nargs);
            return nullptr;
        }

        auto* cpp_bunch =
            parse_bunch(args[0], "The first argument to linac_drift must be of type 'Bunch'");
        if (cpp_bunch == nullptr)
        {
            return nullptr;
        }

        const double length = PyFloat_AsDouble(args[1]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        linac_tracking::linac_drift(cpp_bunch, length);

        Py_RETURN_NONE;
    }

    //Tracking a bunch through a linac quad1 element (linear part of quad tracking) wrapper
    static PyObject* wrap_linac_quad1(PyObject* Py_UNUSED(self), PyObject* const* args,
                                      Py_ssize_t nargs)
    {
        if (nargs != 3)
        {
            PyErr_Format(PyExc_TypeError,
                         "linac_quad1(Bunch, length, kq) takes 3 positional arguments (%zd given).",
                         nargs);
            return nullptr;
        }

        auto* cpp_bunch =
            parse_bunch(args[0], "The first argument to linac_quad1 must be of type 'Bunch'");
        if (cpp_bunch == nullptr)
        {
            return nullptr;
        }

        const double length = PyFloat_AsDouble(args[1]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        const double kq = PyFloat_AsDouble(args[2]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        linac_tracking::linac_quad1(cpp_bunch, length, kq, 1);

        Py_RETURN_NONE;
    }

    //Tracking a bunch through a linac quad2 element (non linear part of quad tracking) wrapper
    static PyObject* wrap_linac_quad2(PyObject* Py_UNUSED(self), PyObject* const* args,
                                      Py_ssize_t nargs)
    {
        if (nargs != 2)
        {
            PyErr_Format(PyExc_TypeError,
                         "linac_quad2(Bunch, length) takes 2 positional arguments (%zd given).",
                         nargs);
            return nullptr;
        }

        auto* cpp_bunch =
            parse_bunch(args[0], "The first argument to linac_quad2 must be of type 'Bunch'");
        if (cpp_bunch == nullptr)
        {
            return nullptr;
        }

        const double length = PyFloat_AsDouble(args[1]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        linac_tracking::linac_quad2(cpp_bunch, length);

        Py_RETURN_NONE;
    }

    //Tracking a bunch through a linac quad3 element (non-linear part of quad tracking) wrapper
    static PyObject* wrap_linac_quad3(PyObject* Py_UNUSED(self), PyObject* const* args,
                                      Py_ssize_t nargs)
    {
        if (nargs != 3)
        {
            PyErr_Format(
                PyExc_TypeError,
                "linac_quad3(Bunch, length, dB_dz) takes 3 positional arguments (%zd given).",
                nargs);
            return nullptr;
        }

        auto* cpp_bunch =
            parse_bunch(args[0], "The first argument to linac_quad3 must be of type 'Bunch'");
        if (cpp_bunch == nullptr)
        {
            return nullptr;
        }

        const double length = PyFloat_AsDouble(args[1]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        const double dB_dz = PyFloat_AsDouble(args[2]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        linac_tracking::linac_quad3(cpp_bunch, length, dB_dz);

        Py_RETURN_NONE;
    }

    //Tracking a bunch through a linac kicker element with different kick for different energies
    static PyObject* wrap_linac_kick(PyObject* Py_UNUSED(self), PyObject* const* args,
                                     Py_ssize_t nargs)
    {
        if (nargs != 4)
        {
            PyErr_Format(PyExc_TypeError, "kick(Bunch, kx, ky, kE) takes 4 arguments (%zd given).",
                         nargs);
            return nullptr;
        }

        auto* cpp_bunch =
            parse_bunch(args[0], "The first argument to kick() must be of type 'Bunch'");
        if (cpp_bunch == nullptr)
        {
            return nullptr;
        }

        const double kx = PyFloat_AsDouble(args[1]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        const double ky = PyFloat_AsDouble(args[2]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        const double kE = PyFloat_AsDouble(args[3]);
        if (PyErr_Occurred())
        {
            return nullptr;
        }

        linac_tracking::kick(cpp_bunch, kx, ky, kE, true);

        Py_RETURN_NONE;
    }

    static PyMethodDef linactrackingMethods[] = {
        {"drift", _PyCFunction_CAST(wrap_linac_drift), METH_FASTCALL,
         "Tracking a bunch through a linac drift "},
        {"quad1", _PyCFunction_CAST(wrap_linac_quad1), METH_FASTCALL,
         "Tracking a bunch through a linear part of a linac quad"},
        {"quad2", _PyCFunction_CAST(wrap_linac_quad2), METH_FASTCALL,
         "Tracking a bunch through a non-linear part of a linac quad"},
        {"quad3", _PyCFunction_CAST(wrap_linac_quad3), METH_FASTCALL,
         "Tracking a bunch through a non-linear part of a linac quad: long.field"},
        {"kick", _PyCFunction_CAST(wrap_linac_kick), METH_FASTCALL,
         "Tracking a bunch through a kicker"},
        {NULL, NULL}};

    static struct PyModuleDef linacTrackingModDef = {PyModuleDef_HEAD_INIT, "linac_tracking",
                                                     "Linac bunch tracking C++ classes", -1,
                                                     linactrackingMethods};

    void initlinactracking(PyObject* module)
    {
        PyObject* module_tr = PyModule_Create(&linacTrackingModDef);
        if (module_tr == nullptr)
        {
            return;
        }

        if (PyModule_AddObject(module, const_cast<char*>("linac_tracking"), module_tr) < 0)
        {
            Py_DECREF(module_tr);
            return;
        }
    }

#ifdef __cplusplus
}
#endif

} // namespace wrap_linac_tracking
