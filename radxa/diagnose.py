"""Read-only discovery: no torque enable, EEPROM write or motor target write."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--port');p.add_argument('--ids',type=int,nargs='+');a=p.parse_args()
    for d in sorted(Path('/sys/bus/iio/devices').glob('iio:device*')):
        print('IIO',d,(d/'name').read_text().strip() if (d/'name').exists() else '?')
        print(' channels:',[x.name for x in d.glob('in_*') if x.name.endswith(('_raw','_scale'))])
    print('UART:',[str(x) for x in Path('/dev').glob('ttyS*')])
    if a.port:
        if not a.ids:p.error('Specify known --ids explicitly; no automatic broad bus scan')
        from feetech_bus import FeetechBus
        bus=FeetechBus(a.port)
        try:
            for sid in a.ids:
                if not 0<=sid<254:raise ValueError('Invalid servo ID')
                try:print(sid,json.dumps(bus.state(sid)))
                except Exception as e:print(sid,'READ FAILED',str(e))
        finally:bus.close()
if __name__=='__main__':main()
