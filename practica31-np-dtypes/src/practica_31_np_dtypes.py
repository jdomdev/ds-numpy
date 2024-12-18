# A Comprehensive Guide to NumPy Data Types
# What else is out there besides int32 and float64?

import numpy as np # -> pip install numpy==1.26.4
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

import numpy as np
import datetime as dt
import pandas as pd # -> pip install pandas==1.4.4
print(pd.__version__) # test
a = np.arange(np.datetime64('2021-01-20'),
                np.datetime64('2021-12-20'),
                np.timedelta64(90, 'D')); a
# = array(['2021-01-20', '2021-04-20', '2021-07-19', '2021-10-17'],
#      dtype='datetime64[D]')

(a.astype('M8[M]') - a.astype('M8[Y]')).view(np.int64)
# = array([0, 3, 6, 9], dtype=int64)




s = pd.DatetimeIndex(a); s
# = DatetimeIndex(['2021-01-20', '2021-04-20', '2021-07-19', '2021-10-17'], 
#       dtype='datetime64[ns]', freq=None)

s.month
# = Int64Index([1, 4, 7, 10], dtype='int64')

def dt2cal(dt):
    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out

dt2cal(a)
# = array([[2021,    1,   20,    0,    0,    0,    0],
#       [2021,    4,   20,    0,    0,    0,    0],
#       [2021,    7,   19,    0,    0,    0,    0],
#       [2021,   10,   17,    0,    0,    0,    0]], dtype=uint32)

np.array(['2020-03-01', '2022-03-01', '2024-03-01'], np.datetime64) - \
    np.array(['2020-02-01', '2022-02-01', '2024-02-01'], np.datetime64)
# = array([29, 28, 29], dtype='timedelta64[D]')

np.datetime64('2016-12-31T23:59:60')
# = ValueError: Seconds out of range in datetime string "2016-12-31T23:59:60"

from astropy.time import Time # -> pip install astropy
(Time('2017-01-01') - Time('2016-12-31 23:59')).sec
# = 61.00000000001593

import numpy as np
np.datetime64('2262-01-01', 'ns') - np.datetime64('1678-01-01', 'ns')
# = numpy.timedelta64(-17537673709551616,'ns')

a = np.arange(np.datetime64('2022-01-01 12:00'),
                np.datetime64('2022-01-03 12:00'),
                np.timedelta64(1, 'D'))
np.datetime_as_string(a)
# = array(['2022-01-01T12:00', '2022-01-02T12:00'], dtype='<U35')

np.datetime_as_string(a, timezone='local')
# = array(['2022-01-01T13:00+0100', '2022-01-02T13:00+0100'], dtype='<U39')

import pytz as pytz # pip install pytz==2022.6
np.datetime_as_string(a, timezone=pytz.timezone('US/Eastern'))
# = array(['2022-01-01T07:00-0500', '2022-01-02T07:00-0500'], dtype='<U39')



# 6. COMBINATIONS THEREOF
#       (Combinaciones de los mismos)
# ------------------------------------

a = np.array([[3,4], [2,7], [1,5], [2,4]]); a
# = array([[3, 4],
#       [2, 7],
#       [1, 5],
#       [2, 4]])

# Ejemplo creado con función 'u2s' creada:
#   (No consigo que la salida sea igual)!!
# Definición de las funciones u2s y s2u

import numpy as np

def u2s(array):
    structured_array = np.array(array, dtype=[('f0', '<i4'), ('f1', '<i4')])
    return structured_array

def s2u(array):
    unstructured_array = array.view(np.int32).reshape(array.shape + (-1,))
    return unstructured_array

a = np.array([[3,4], [2,7], [1,5], [2,4]]); a
b = u2s(a); b

# = array([[3, 4],
#       [2, 7],
#       [1, 5],
#       [2, 4]])

# = array([[(3, 3), (4, 4)],
#       [(2, 2), (7, 7)],
#       [(1, 1), (5, 5)],
#       [(2, 2), (4, 4)]], dtype=[('f0', '<i4'), ('f1', '<i4')])

b.sort(order=['f1', 'f0']); b

s2u(b)


np.genfromtxt('pract31-dtypes/data/a.csv', dtype=None, encoding=None, delimiter=', ', names=True)
# = array([('John', 21, 1.77,  True), ('Mary', 20, 1.63, False)],
#      dtype=[('name', '<U4'), ('age', '<i8'), ('height', '<f8'), ('is_married', '?')])

a = np.array([('John', 21, 1.77, True),
               ('Mary', 20, 1.63, False)])
np.core.records.fromarrays(zip(*a))
# = rec.array([('John', '21', '1.77', 'True'),
#           ('Mary', '20', '1.63', 'False')],
#          dtype=[('f0', '<U4'), ('f1', '<U2'), ('f2', '<U4'), ('f3', '<U5')])

a = np.array([('John', 21, 1.77, True),
               ('Mary', 20, 1.63, False)],
        dtype=[('name', 'U4'), ('age', int), ('height', float), ('is_married', bool)])
np.core.records.fromarrays([
      np.array(['John', 'Mary']),
      np.array([21, 20]),
      np.array([1.77, 1.63]),
      np.array([True, False])])
# = rec.array([('John', 21, 1.77,  True), ('Mary', 20, 1.63, False)],
#          dtype=[('f0', '<U4'), ('f1', '<i8'), ('f2', '<f8'), ('f3', '?')])

rgb = np.dtype([('x', np.uint8), ('y', np.uint8), ('z', np.uint8)])
a = np.zeros(5, dtype=rgb); a
# = array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
#      dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])

a[0]
# = (0, 0, 0)
a[0]['x']
# = 0
a[0]['x'] = 10
a
# = array([(10, 0, 0), ( 0, 0, 0), ( 0, 0, 0), ( 0, 0, 0), ( 0, 0, 0)],
#           dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])
a['z'] = 5
a
# = array([(10, 0, 5), ( 0, 0, 5), ( 0, 0, 5), ( 0, 0, 5), ( 0, 0, 5)],
#      dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])

b = a.view(np.recarray)
b
# = rec.array([(10, 0, 5), ( 0, 0, 5), ( 0, 0, 5), ( 0, 0, 5), ( 0, 0, 5)],
#               dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])
b[0].x
# = 10
b.y=7; b
# = rec.array([(10, 7, 5), ( 0, 7, 5), ( 0, 7, 5), ( 0, 7, 5), ( 0, 7, 5)],
#               dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1')])


a = np.random.rand(100000, 2)

b = a.view(dtype=[('x', np.float64), ('y', np.float64)])

c = np.recarray(buf=a, shape=len(a), dtype=
                [('x', np.float64), ('y', np.float64)])

s1 = 0
for r in a:
    s1 += (r[0]**2 + r[1]**2)**-1.5          # reference

s2 = 0
for r in b:
    s2 += (r['x']**2 + r['y']**2)**-1.5      # 5x slower

s3 = 0
for r in c:
    s3 += (r.x**2 + r.y**2)**-1.5            # 7x slower

S1 = np.sum((a[:, 0]**2 + a[:, 1]**2)**-1.5) # 20x faster
S2 = np.sum((b['x']**2 + b['y']**2)**-1.5)   # same as S1
S3 = np.sum((c.x**2 + c.y**2)**-1.5)         # same as S1


# 7. TYPE CHECKS
#       (COMPROBACIONES DE TIPO)
# ----------------


a = np.array([1, 2, 3], dtype=np.int32)  # Especificar explícitamente el tipo de datos
v = a[0]
isinstance(v, np.int32)    # might be np.int64 on a different OS
# = True

isinstance(v, np.integer)        # true for all integers
# = True
isinstance(v, np.number)         # true for integers and floats
# = True
isinstance(v, np.floating)       # true for floats except complex
# = False
isinstance(v, np.complexfloating) # true for complex floats only
# = False

a.dtype == np.int32
# = True
a.dtype == np.int64
# = False

x.dtype in (np.half, np.single, np.double, np.longdouble)
#False # -> NO COMPROBADO

np.issubdtype(a.dtype, np.integer)
# = True
np.issubdtype(a.dtype, np.floating)
# = False


pd.api.types.is_integer_dtype(a.dtype)
# = True
pd.api.types.is_float_dtype(a.dtype)
# = False

np.typecodes
# = {'Character': 'c', 'Integer': 'bhilqp', 'UnsignedInteger': 'BHILQP', 
#   'Float': 'efdg', 'Complex': 'FDG', 'AllInteger': 'bBhHiIlLqQpP', 
#   'AllFloat': 'efdgFDG', 'Datetime': 'Mm', 'All': '?bhilqpBHILQPefdgFDGSUVOMm'}

a.dtype.char in np.typecodes['AllInteger']
# = True
a.dtype.char in np.typecodes['Datetime']
# = False

