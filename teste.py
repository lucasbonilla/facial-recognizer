import os
import Utils as ut

for a, b, c in os.walk(ut.IMAGESPATH, topdown=False):
    print(a)
    print(b)
    print(c)

