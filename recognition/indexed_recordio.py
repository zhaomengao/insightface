import mxnet as mx
from mxnet.recordio import *

_IR_FORMAT_64 = 'IdQQ'
_IR_SIZE_64= struct.calcsize(_IR_FORMAT_64)

def pack64(header, s):
    header = IRHeader(*header)
    if isinstance(header.label, numbers.Number):
        header = header._replace(flag=0)
    else:
        label = np.asarray(header.label, dtype=np.float64)
        header = header._replace(flag=label.size, label=0)
        s = label.tostring() + s
    s = struct.pack(_IR_FORMAT_64, *header) + s
    return s

def unpack64(s, multi_record=False):
    if multi_record:
        q = Queue.Queue()
        begin = 0
        while begin < len(s):
            header_t = IRHeader(*struct.unpack(_IR_FORMAT_64, s[begin:_IR_SIZE_64]))
            end = begin + _IR_SIZE_64 + header_t.flag
            q.put((header_t, s[begin+_IR_SIZE_64: end]))
            begin = end
        return q

    header = IRHeader(*struct.unpack(_IR_FORMAT_64, s[:_IR_SIZE_64]))
    s = s[_IR_SIZE_64:]
    if header.flag > 0:
        header = header._replace(label=np.frombuffer(s, np.float64, header.flag))
        s = s[header.flag*8:]
    return header, s
