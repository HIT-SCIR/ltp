#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

class Error(Exception):
    pass


class RegistrationError(Error):
    pass


class DataUnsupported(Error):
    pass
