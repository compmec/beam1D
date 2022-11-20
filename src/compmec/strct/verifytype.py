import functools
from typing import Any, Optional

import numpy as np


def raise_type_error(namevar: str, typeshouldbe: str, currenttype: str):
    error_msg = f"Variable `{str(namevar)}` should be type ({typeshouldbe}) "
    error_msg += f"but instead is type ({currenttype})"
    raise TypeError(error_msg)


class Float(object):
    @classmethod
    def verify(cls, var: float, namevar: str):
        if not isinstance(var, (int, float)):
            typeshouldbe = str(cls).replace("compmec.strct.verifytype.", "")
            currenttype = str(type(var))
            raise_type_error(namevar, typeshouldbe, currenttype)


class PositiveFloat(Float):
    @classmethod
    def verify(cls, value: float, namevar: str):
        Float.verify(value, namevar)
        if value <= 0:
            error_msg = (
                f"Variable `{namevar}` should be positive but instead it's {value}"
            )
            raise ValueError(error_msg)


def type_check(func):
    @functools.wraps(func)
    def check(*args, **kwargs):
        # if args[0].hasattr(func)
        print(args)
        print(args[0])
        print(func)
        print(type(args))
        print(type(func))
        print(dir(func))
        print(func.__doc__)
        print(func.__module__)
        print(func.__annotations__)
        print(func.__code__)
        print(func.__closure__)
        print(func.__call__)

        for i in range(len(args)):
            v = args[i]
            v_name = list(func.__annotations__.keys())[i]
            v_type = list(func.__annotations__.values())[i]
            if hasattr(v_type, "verify"):
                v_type.verify(v, v_name)
            else:
                error_msg = "Variable `" + str(v_name) + "` should be type ("
                error_msg += (
                    str(v_type) + ") but instead is type (" + str(type(v)) + ")"
                )
                if not isinstance(v, v_type):
                    raise TypeError(error_msg)
        result = func(*args, **kwargs)
        if "return" in func.__annotations__:
            v = result
            v_name = "return"
            v_type = func.__annotations__["return"]
            if v_type is None:
                if v is not None:
                    error_msg = f"Variable `return` should be type (None) "
                    error_msg += f"but instead is type ({str(type(v))})"
                    raise TypeError(error_msg)
            elif hasattr(v_type, "verify"):
                v_type.verify(v, v_name)
            else:
                error_msg = f"Variable `return` should be type ({str(v_type)}) "
                error_msg += f"but instead is type ({str(type(v))})"
                if not isinstance(v, v_type):
                    raise TypeError(error_msg)
        return result

    return check
