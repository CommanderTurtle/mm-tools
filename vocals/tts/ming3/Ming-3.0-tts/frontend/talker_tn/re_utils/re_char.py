# Copyright (c) Ant Group. All rights reserved.
from .re_base import REBase


class REChar(REBase):
    name = 'char'
    
    def __init__(self, txt):
        super().__init__(txt)

    def to_string(self):
        return self.txt