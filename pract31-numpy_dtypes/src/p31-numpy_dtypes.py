# A Comprehensive Guide to NumPy Data Types
# What else is out there besides int32 and float64?

import numpy as np
print(np.__version__) # test


# 1.INTEGERS
# ----------


np.array([1,2,3]).dtype
# dtype('int32')

np.arange(10**6)
# array([     0,      1,      2, ..., 999997, 999998, 999999])
np.arange(10**6).reshape(1000, 1000)
'''
array([[     0,      1,      2, ...,    997,    998,    999],
       [  1000,   1001,   1002, ...,   1997,   1998,   1999],
       [  2000,   2001,   2002, ...,   2997,   2998,   2999],
       ...,
       [997000, 997001, 997002, ..., 997997, 997998, 997999],
       [998000, 998001, 998002, ..., 998997, 998998, 998999],
       [999000, 999001, 999002, ..., 999997, 999998, 999999]])
'''

np.array([255], np.uint8) + 1
# array([0], dtype=uint8)

np.array([2**31-1])
# array([2147483647])

np.array([2**31-1]) + 1
# array([-2147483648])

np.array([2**63-1]) + 1
# array([-9223372036854775808], dtype=int64)

np.array([255], np.uint8)[0] + 1
# 256

np.array([2**31-1])[0] + 1
# -2147483648

np.array([2**63-1])[0] + 1
# <stdin>:1: RuntimeWarning: overflow encountered in scalar add
# -9223372036854775808
import numpy as np
with np.errstate(over='raise'):
    print(np.array([2**31-1])[0]+1)
# FloatingPointError: overflow encountered in scalar add

with np.errstate(over='ignore'):
    print(np.array([2**31-1])[0]+1)
# -2147483648

a = np.array([10], dtype=object)
len(str(a**1000))
# 1003


# 2. FLOATS
# ---------


x = np.array([-1234.5])
1/(1+np.exp(-x))
# <stdin>:1: RuntimeWarning: overflow encountered in exp
# array([0.])

np.exp(np.array([1234.5]))
# array([inf])

x = np.array([-1234.5], dtype=np.float128)
1/(1+np.exp(-x))
# array([0.]) # -> float128: no disponible en Windows

9279945539648888.0+1
# = 9279945539648888.0

len('9279945539648888')
# = 16

from decimal import Decimal as D
a = np.array([D('0.1'), D('0.2')]); a
# = array([Decimal('0.1'), Decimal('0.2')], dtype=object)
a.sum()
# = Decimal('0.3')

from fractions import Fraction
a = np.array([Fraction(1, 10), Fraction(1, 5)], dtype=object)
# = a = np.array([Fraction(1, 10), Fraction(1, 5)], dtype=object)
a.sum()
# = Fraction(3, 10)

np.array([1+2j])
# = array([1.+2.j])

a = np.array([np.nan, 5, np.nan, 5, 5])
a[~np.isnan(a)].mean()
# = 5.0


# 3. BOOLS
# --------

import sys
sys.getsizeof(True)
# = 28


# 4. CADENAS
# ----------

np.array(['abcde', 'x', 'y', 'x'])
# = array(['abcde', 'x', 'y', 'x'], dtype='<U5')


np.array(['abcde', 'x', 'y', 'x'])
# = array(['abcde', 'x', 'y', 'x'], dtype='<U5')

np.array(['abcde', 'x', 'y', 'x'], object)
# = array(['abcde', 'x', 'y', 'x'], dtype=object)

np.array([b'abcde', b'x', b'y', b'x'])
# = array([b'abcde', b'x', b'y', b'x'], dtype='|S5')

np.char.upper(np.array([['a','b'],['c','d']]))
# = array([['A', 'B'],
#       ['C', 'D']], dtype='<U1')

a = np.array([['a','b'],['c','d']], object)
np.vectorize(lambda x: x.upper(), otypes=[object])(a)
# = array([['A', 'B'],
#       ['C', 'D']], dtype=object)


# 5. DATETIMES
# ------------

np.datetime64('today')
# = numpy.datetime64('2024-06-20')

np.datetime64('now')
# = numpy.datetime64('2024-06-20T18:03:45')

import datetime as dt
dt = dt.datetime.utcnow()
np.datetime64(dt)
# = numpy.datetime64('2024-06-20T18:11:55.780089')

np.datetime64('2024-06-20 18:11:55.780089836')
# = numpy.datetime64('2024-06-20T18:11:55.780089836')

np.datetime64(dt.utcnow(), 'ns')
# = numpy.datetime64('2024-06-20T18:19:32.911035000')

a = np.array([dt.utcnow()], dtype='datetime64[100ms]'); a
# = array(['2024-06-20T18:21:52.800'], dtype='datetime64[100ms]')

a + 1
# = array(['2024-06-20T18:21:52.900'], dtype='datetime64[100ms]')

a[0].dtype
# = dtype('<M8[100ms]')

np.datetime_data(a[0])
# = ('ms', 100)

z = np.datetime64('2022-01-01') - np.datetime64(dt.now()); z
# = numpy.timedelta64(-77921524649541,'us')

z.item() 
# = datetime.timedelta(days=-902, seconds=11275, microseconds=350459)

z.item().total_seconds()
# = -77921524.649541

np.datetime64('2022-01-01') - np.datetime64(dt.now(), 's')
# = numpy.timedelta64(-77920458,'s')

np.datetime64('2021-12-24 18:14:23').item()
# = datetime.datetime(2021, 12, 24, 18, 14, 23)

np.datetime64('2021-12-24 18:14:23').item().month
# = 12

a = np.arange(np.datetime64('2021-01-20'),
                np.datetime64('2021-12-20'),
                np.timedelta64(90, 'D')); a
# = array(['2021-01-20', '2021-04-20', '2021-07-19', '2021-10-17'],
#      dtype='datetime64[D]')

(a.astype('M8[M]') - a.astype('M8[Y]')).view(np.int64)
# = array([0, 3, 6, 9], dtype=int64)




#import pandas as pd
#s = pd.DatetimeIndex(a); s
