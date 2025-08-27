import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/pedroromero/mapping_eracing/dv_stack/install/dv_stack'
