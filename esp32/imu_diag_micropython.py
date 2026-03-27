from machine import Pin, SoftI2C
from time import sleep_ms

SDA_PIN = 41
SCL_PIN = 42
I2C_FREQ = 100000

ADDR_CANDIDATES = (0x68, 0x69)

REG_WHO_AM_I = 0x75
REG_PWR_MGMT_1 = 0x6B


def read_whoami(i2c, addr):
    try:
        # Wake up MPU6050 if it is sleeping.
        i2c.writeto_mem(addr, REG_PWR_MGMT_1, bytes([0x00]))
        sleep_ms(20)
        data = i2c.readfrom_mem(addr, REG_WHO_AM_I, 1)
        return data[0]
    except Exception as exc:
        print("read_whoami fallo en 0x{:02X}: {}".format(addr, exc))
        return None


def main():
    print("=== IMU MicroPython diag ===")
    print("Cableado esperado:")
    print("  VCC -> 3V3")
    print("  GND -> GND")
    print("  SDA -> GPIO41")
    print("  SCL -> GPIO42")

    i2c = SoftI2C(
        scl=Pin(SCL_PIN, Pin.OPEN_DRAIN, Pin.PULL_UP),
        sda=Pin(SDA_PIN, Pin.OPEN_DRAIN, Pin.PULL_UP),
        freq=I2C_FREQ,
    )
    print("Bus I2C creado en SDA={} SCL={} freq={}".format(SDA_PIN, SCL_PIN, I2C_FREQ))

    while True:
        try:
            devices = i2c.scan()
            print("scan ->", [hex(d) for d in devices])
        except Exception as exc:
            print("scan fallo:", exc)
            devices = []

        found = False
        for addr in ADDR_CANDIDATES:
            who = read_whoami(i2c, addr)
            if who is not None:
                print("addr=0x{:02X} WHO_AM_I=0x{:02X}".format(addr, who))
                found = True

        if not found:
            print("Sin respuesta valida en 0x68 ni 0x69")

        print("---")
        sleep_ms(2000)


main()
