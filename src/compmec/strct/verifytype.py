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
