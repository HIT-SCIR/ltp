#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>
# 参数弃用

def deprecated(info, force=False):
    import sys

    def decorator(func):
        def func_wrapper(*args, **kwargs):
            if force:
                raise NotImplementedError(info)
            print('弃用警告: ', info, file=sys.stderr)
            return func(*args, **kwargs)

        return func_wrapper

    return decorator


class deprecated_param(object):
    def __init__(self, deprecated_args, version, reason):
        self.deprecated_args = set(deprecated_args.split())
        self.version = version
        self.reason = reason

    def __call__(self, callable):
        def wrapper(*args, **kwargs):
            found = self.deprecated_args.intersection(kwargs)
            if found:
                raise TypeError("Parameter(s) %s deprecated since version %s; %s" % (
                    ', '.join(map("'{}'".format, found)), self.version, self.reason))
            return callable(*args, **kwargs)

        return wrapper
