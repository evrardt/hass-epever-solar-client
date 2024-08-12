import sys
import datetime
import os
import json
import logging
import ctypes
import threading
import signal
import time
from datetime import timedelta

import paho.mqtt.client as mqtt

from pymodbus.client import ModbusTcpClient
from pymodbus.transaction import ModbusRtuFramer as ModbusFramer
from pymodbus.mei_message import ReadDeviceInformationRequest

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

logger: logging.Logger = logging.getLogger('epever-solar-client')

# --------------------------------------------------------------


def value32(low, high):
    return ctypes.c_int(low + (high << 16)).value / 100


def value16(value):
    return ctypes.c_short(value).value / 100


def value8(value):
    return [value >> 8, value & 0xFF]


def toBool(value):
    values = {
        0: False,
        1: True
    }
    return values.get(value, None)


def days(value):
    return "{} Days".format(value)


def minutes(value):
    return "{} Minutes".format(value)


def seconds(value):
    return "{} Seconds".format(value)


def hourMinute(value):
    hm = value8(value)
    return "{0} hours {1} minutes".format(hm[0], hm[1])


def getTime(second, minute, hour):
    return datetime.time(hour, minute, second)


def toFloat(low, high=None):
    if high is None:
        return value16(low)
    else:
        return value32(low, high)
# --------------------------------------------------------------


def log_info(info, *arguments):
    if len(arguments) == 0:
        logger.info(info)
    elif len(arguments) == 1:
        logger.info(info + "%s" % arguments)
    else:
        logger.info(info + ' %r', arguments)


class EpeverSolarClient:
    _host = "192.168.88.78"
    _port = 9999

    def __init__(self, host, port):
        self._host = host
        self._port = port

    def get_device_info(self, client):
        request = ReadDeviceInformationRequest(slave=1)
        response = client.execute(request)
        if (not response.isError()):
            result = []
            for idx in response.information:
                result.append(response.information[idx].decode("ascii"))
            return result
        else:
            logger.error(response)
            return None

    def init_clock(self, client):
        print("Updating Device RTC...")
        result = client.read_holding_registers(0x9013, 3, slave=1)
        if (not result.isError()):
            print("... Date({}, {}, {}, {}, {}, {})".format(
                2000 + (result.registers[2] >> 8),
                result.registers[2] & 0xFF,
                result.registers[1] >> 8,
                result.registers[1] & 0xFF,
                result.registers[0] >> 8,
                result.registers[0] & 0xFF
            ))

            now = datetime.now()
            newData = [0, 0, 0]
            newData[2] = ((now.year - 2000) << 8) + now.month
            newData[1] = (now.day << 8) + now.hour
            newData[0] = (now.minute << 8) + now.second
            print("... Date({}, {}, {}, {}, {}, {})".format(
                2000 + (newData[2] >> 8),
                newData[2] & 0xFF,
                newData[1] >> 8,
                newData[1] & 0xFF,
                newData[0] >> 8,
                newData[0] & 0xFF
            ))
            result = client.write_registers(0x9013, newData, slave=1)
            if (not result.isError()):
                print("Err:", "Updating Device RTC")
            else:
                print("Device RTC Updated.")

    def init_settings(self, client, settings):
        print("Configuring battery setting...")
        result = client.write_registers(0x9000, settings, slave=1)
        if (not result.isError()):
            print("Battery setting done.")
        else:
            print("Err:", "Battery setting")

    def configure_battery(self, client):
        # Bosch T5 077 6-СТ 180Ah L+ 1000A 0092T50770
        # 12v / 180 Ah

        settings = [
            0,       # 9000 Battery Type 0 =  User                             #1
            0x00B4,  # 9001 Battery Cap B4 == 180AH                            #2
            0x012C,  # 9002 Temp compensation -3V /°C/2V                       #3
            0x05DC,  # 9003 0x5DC == 1500 Over Voltage Disconnect Voltage 15,0 #4
            0x058C,  # 9004 0x58C == 1480 Charging Limit Voltage 14,8          #5
            0x058C,  # 9005 Over Voltage Reconnect Voltage 14,8                #6
            0x05BF,  # 9006 Equalize Charging Voltage 14,6                     #7
            0x05BE,  # 9007 Boost Charging Voltage 14,7                        #8
            0x0550,  # 9008 Float Charging Voltage 13,6                        #9
            0x0528,  # 9009 Boost Reconnect Charging Voltage 13,2              #10
            0x04C4,  # 900A Low Voltage Reconnect Voltage 12,2                 #1
            0x04B0,  # 900B Under Voltage Warning Reconnect Voltage 12,0       #12
            0x04BA,  # 900c Under Volt. Warning Volt 12,1                      #13
            0x04BA,  # 900d Low Volt. Disconnect Volt. 11.8                    #14
            0x04BA   # 900E Discharging Limit Voltage 11.8                     #15
        ]

        self.init_settings(client, settings)

    def get_battery_settings(self, client):
        settings = {}
        result = client.read_holding_registers(0x9000, 15, slave=1)
        if (not result.isError()):
            batteryType = {
                0: "User defined",
                1: "Sealed",
                2: "GEL",
                3: "Flooded"
            }
            settings["batteryType"] = batteryType.get(result.registers[0])
            settings["batteryCapacity"] = result.registers[1]
            settings["temperatureCompensationCoefficient"] = toFloat(
                result.registers[2])
            settings["highVoltDisconnect"] = toFloat(result.registers[3])
            settings["chargingLimitVoltage"] = toFloat(result.registers[4])
            settings["overVoltageReconnect"] = toFloat(result.registers[5])
            settings["equalizationVoltage"] = toFloat(result.registers[6])
            settings["boostVoltage"] = toFloat(result.registers[7])
            settings["floatVoltage"] = toFloat(result.registers[8])
            settings["boostReconnectVoltage"] = toFloat(result.registers[9])
            settings["lowVoltageReconnect"] = toFloat(result.registers[10])
            settings["underVoltageRecover"] = toFloat(result.registers[11])
            settings["underVoltageWarning"] = toFloat(result.registers[12])
            settings["lowVoltageDisconnect"] = toFloat(result.registers[13])
            settings["dischargingLimitVoltage"] = toFloat(
                result.registers[14])
        else:
            logger.error(result)
            return None

        return settings

    def get_battery_load(self, client):
        result = client.read_input_registers(0x310C, 4, slave=1)
        if (not result.isError()):
            loadVoltage = toFloat(result.registers[0])
            loadCurrent = toFloat(result.registers[1])
            loadPower = toFloat(result.registers[2], result.registers[3])
            all = {}
            all["loadVoltage"] = loadVoltage
            all["loadCurrent"] = loadCurrent
            all["loadPower"] = loadPower

            return all
        else:
            logger.error(result)
            return None

    def get_data(self, client):
        data = {}

        result = client.read_input_registers(0x3100, 19, slave=1)
        if (not result.isError()):
            data["chargingInputVoltage"] = toFloat(result.registers[0])
            data["chargingInputCurrent"] = toFloat(result.registers[1])
            data["chargingInputPower"] = toFloat(
                result.registers[2], result.registers[3])

            data["chargingOutputVoltage"] = toFloat(result.registers[4])
            data["chargingOutputCurrent"] = toFloat(result.registers[5])
            data["chargingOutputPower"] = toFloat(
                result.registers[6], result.registers[7])

            data["dischargingOutputVoltage"] = toFloat(result.registers[12])
            data["dischargingOutputVurrent"] = toFloat(result.registers[13])
            data["dischargingOutputPower"] = toFloat(
                result.registers[14], result.registers[15])

            data["batteryTemperature"] = toFloat(result.registers[16])
            data["temperatureInside"] = toFloat(result.registers[17])
            data["powerComponentsTemperature"] = toFloat(result.registers[18])
        else:
            logger.error(result)
            return None

        result = client.read_input_registers(0x311A, 2, slave=1)
        if (not result.isError()):
            data["batterySoC"] = toFloat(result.registers[0]) * 100
            data["remoteBatteryTemperature"] = toFloat(result.registers[1])
        else:
            logger.error(result)
            return None

        result = client.read_input_registers(0x311D, 1, slave=1)
        if (not result.isError()):
            data["batteryRealRatedPower"] = toFloat(result.registers[0])
        else:
            logger.error(result)
            return None

        return data

    def get_battery_stat(self, client):
        result = client.read_input_registers(0x3300, 31, slave=1)
        if (not result.isError()):
            stat = {}
            stat["maxVoltToday"] = toFloat(result.registers[0])
            stat["minVoltToday"] = toFloat(result.registers[1])
            stat["maxBatteryVoltToday"] = toFloat(result.registers[2])
            stat["minBatteryVoltToday"] = toFloat(result.registers[3])
            stat["consumedEnergyToday"] = toFloat(
                result.registers[4], result.registers[5])
            stat["consumedEnergyMonth"] = toFloat(
                result.registers[6], result.registers[7])
            stat["consumedEnergyYear"] = toFloat(
                result.registers[8], result.registers[9])
            stat["totalConsumedEnergy"] = toFloat(
                result.registers[10], result.registers[11])
            stat["generatedEnergyToday"] = toFloat(
                result.registers[12], result.registers[13])
            stat["generatedEnergyMonth"] = toFloat(
                result.registers[14], result.registers[15])
            stat["generatedEnergyYear"] = toFloat(
                result.registers[16], result.registers[17])
            stat["totalGeneratedEnergy"] = toFloat(
                result.registers[18], result.registers[19])
            stat["carbonDioxideReduction"] = toFloat(
                result.registers[20], result.registers[21])
            stat["batteryVoltage"] = toFloat(result.registers[26])
            stat["batteryCurrent"] = toFloat(
                result.registers[27], result.registers[28])
            stat["batteryTemperature"] = toFloat(result.registers[29])
            stat["ambientTemperature"] = toFloat(result.registers[30])
            return stat
        else:
            logger.error(result)
            return None

    def get_battery_status(self, client):
        result = client.read_input_registers(0x3200, 3, slave=1)
        if (not result.isError()):
            value = result.registers[0]
            batteryStatusVoltage = {
                0: "Normal",
                1: "Overvolt",
                2: "Under volt",
                3: "Low Volt Disconnect",
                4: "Fault"
            }
            batteryStatusTemperature = {
                0: "Normal",
                1: "Over Temperature",
                2: "Low Temperature"
            }
            abnormalStatus = {
                0: "Normal",
                1: "Abnormal"
            }
            wrongStatus = {
                0: "Correct",
                1: "Wrong"
            }
            batteryStatus = {}
            batteryStatus["voltage"] = batteryStatusVoltage.get(
                value & 0x0007, None)
            batteryStatus["temperature"] = batteryStatusTemperature.get(
                (value >> 4) & 0x000f, None)
            batteryStatus["internalResistance"] = abnormalStatus.get(
                (value >> 8) & 0x0001, None)
            batteryStatus["ratedVoltage"] = wrongStatus.get(
                (value >> 15) & 0x0001, None)

            value = result.registers[1]
            chargingEquipmentStatusInputVoltage = {
                0: "Normal",
                1: "No power connected",
                2: "Higher Volt Input",
                3: "Input Volt Error"
            }
            chargingEquipmentStatusBattery = {
                0: "Not charging",
                1: "Float",
                2: "Boost",
                3: "Equalization"
            }
            faultStatus = {
                0: "Normal",
                1: "Fault"
            }
            runningStatus = {
                0: "Standby",
                1: "Running"
            }

            equipmentStatus = {}
            equipmentStatus["inputVoltage"] = chargingEquipmentStatusInputVoltage.get(
                (value >> 14) & 0x0003, None)
            equipmentStatus["mosfetShort"] = toBool((value >> 13) & 0x0001)
            equipmentStatus["chargingAntiReverseMosfetShort"] = toBool(
                (value >> 12) & 0x0001)
            equipmentStatus["antiReverseMosfetShort"] = toBool(
                (value >> 11) & 0x0001)
            equipmentStatus["inputOverCurrent"] = toBool(
                (value >> 10) & 0x0001)
            equipmentStatus["loadOverCurrent"] = toBool((value >> 9) & 0x0001)
            equipmentStatus["loadShort"] = toBool((value >> 8) & 0x0001)
            equipmentStatus["loadMosfetShort"] = toBool((value >> 7) & 0x0001)
            equipmentStatus["pvInputShort"] = toBool((value >> 4) & 0x0001)
            equipmentStatus["battery"] = chargingEquipmentStatusBattery.get(
                (value >> 2) & 0x0003, None)
            equipmentStatus["fault"] = faultStatus.get(
                (value >> 1) & 0x0001, None)
            equipmentStatus["running"] = runningStatus.get(
                (value) & 0x0001, None)

            value = result.registers[2]
            dischargingEquipmentStatusVoltage = {
                0: "Normal",
                1: "Low",
                2: "High",
                3: "No access input volt error"
            }
            dischargingEquipmentStatusOutput = {
                0: "Light Load",
                1: "Moderate",
                2: "Rated",
                3: "Overload"
            }
            dischargingEquipmentStatus = {}

            dischargingEquipmentStatus["inputVoltage"] = dischargingEquipmentStatusVoltage.get(
                (value >> 14) & 0x0003, None)
            dischargingEquipmentStatus["outputPower"] = dischargingEquipmentStatusOutput.get(
                (value >> 12) & 0x0003, None)
            dischargingEquipmentStatus["shortCircuit"] = toBool(
                (value >> 11) & 0x0001)
            dischargingEquipmentStatus["unableDischarge"] = toBool(
                (value >> 10) & 0x0001)
            dischargingEquipmentStatus["unableStopDischarging"] = toBool(
                (value >> 9) & 0x0001)
            dischargingEquipmentStatus["outputVoltageAbnormal"] = toBool(
                (value >> 8) & 0x0001)
            dischargingEquipmentStatus["inputOverpressure"] = toBool(
                (value >> 7) & 0x0001)
            dischargingEquipmentStatus["highVoltageSideShortCircuit"] = toBool(
                (value >> 6) & 0x0001)
            dischargingEquipmentStatus["boostOverpressure"] = toBool(
                (value >> 5) & 0x0001)
            dischargingEquipmentStatus["outputOverpressure"] = toBool(
                (value >> 4) & 0x0001)

            dischargingEquipmentStatus["fault"] = faultStatus.get(
                (value >> 1) & 0x0001, None)
            dischargingEquipmentStatus["running"] = runningStatus.get(
                value & 0x0001, None)

            return {"batteryStatus": batteryStatus, "equipmentStatus": equipmentStatus, "dischargingEquipmentStatus": dischargingEquipmentStatus}
        else:
            logger.error(result)
            return None

    def run(self, method='get_data'):
        client = ModbusTcpClient(
            self._host, port=self._port, framer=ModbusFramer, retries=5)

        log_info("Running method {}".format(method))
        try:
            success = client.connect()
            if not success:
                log_info("Cannot connect to Epever server")
                return None

            client.send(bytes.fromhex("20020000"))
            run_method = getattr(EpeverSolarClient, method)
            return run_method(self, client)

        except BaseException as e:
            logger.error(e)

        finally:
            client.close()


class Configurator:
    config = {}

    def __init__(self, config):
        self.config = config

    def get(self, path, default=None):
        items = path.split(':')
        value = None

        for idx, item in enumerate(items):
            if idx == 0:
                value = self.config.get(item)
            else:
                value = None if value is None else value.get(item)

        return value or default


class ProgramKilled(Exception):
    pass


class Job(threading.Thread):
    def __init__(self, interval, *args, **kwargs):
        threading.Thread.__init__(self)
        self.daemon = False
        self.stopped = threading.Event()
        self.interval = interval
        self.args = args
        self.kwargs = kwargs
        self.init = True

    def publish(self, publisher, event_name, event):
        base_topic = publisher._userdata.get("base_topic")
        publisher.publish(base_topic+"/" + event_name,
                          payload=json.dumps(event))
        log_info("EVENT:", event)

    def execute(self, publisher, epever_client):
        if self.init:
            data = epever_client.run("get_device_info")
            if data is not None:
                self.publish(publisher, "info", data)
            else:
                logger.error("get_device_info")

            data = epever_client.run("get_battery_settings")
            if data is not None:
                self.publish(publisher, "settings", data)
            else:
                logger.error("get_battery_settings")

        data = epever_client.run("get_data")
        if data is not None:
            self.publish(publisher, "data", data)
        else:
            logger.error("get_data")

        if datetime.datetime.now().minute % 5 == 0 or self.init:
            data = epever_client.run("get_battery_stat")
            if data is not None:
                self.publish(publisher, "stat", data)
            else:
                logger.error("get_battery_stat")

            data = epever_client.run("get_battery_status")
            if data is not None:
                self.publish(publisher, "status", data)
            else:
                logger.error("get_battery_status")

            data = epever_client.run("get_battery_load")
            if data is not None:
                self.publish(publisher, "load", data)
            else:
                logger.error("get_battery_load")

        if self.init:
            self.init = False

    def stop(self):
        self.stopped.set()
        self.join()

    def run(self):
        self.execute(*self.args, **self.kwargs)
        # looping
        while not self.stopped.wait(self.interval.total_seconds()):
            self.execute(*self.args, **self.kwargs)


WAIT_TIME_SECONDS = 60


def signal_handler(signum, frame):
    raise ProgramKilled


def on_connect(client, userdata, flags, rc):
    base_topic = userdata.get("base_topic")
    client.subscribe(base_topic + "/" + "event" + "/#")


def on_message(client, userdata, msg):
    request = msg.payload.decode("ascii")
    event_name = request.split("_")[-1]
    epever_client = userdata.get("epever_client")

    data = epever_client.run(request)
    if data is not None:        
        base_topic = userdata.get("base_topic")
        client.publish(base_topic+"/" + event_name,
                          payload=json.dumps(data))
    else:
        logger.error(str(msg.payload))

    # print(msg.topic+" "+str(msg.payload))


def main():
    log_info("Epever solar client started...")

    HOST, PORT = '0.0.0.0', 15002

    MQTT_HOST = "localhost"
    MQTT_PORT = 1883
    MQTT_USER = ""
    MQTT_PASSWD = ""

    config = None
    if os.path.exists('./epever-solar-client.json'):
        log_info('Running in local mode')
        fp = open('./epever-solar-client.json', 'r')
        config = Configurator(json.load(fp))
        fp.close()
    elif os.path.exists('/data/options.json'):
        log_info('Running in hass.io add-on mode')
        fp = open('/data/options.json', 'r')
        config = Configurator(json.load(fp))
        fp.close()
    else:
        log_info('Configuration file not found, exiting.')
        sys.exit(1)

    if config.get('mqtt:debug'):
        log_info("Debugging messages enabled")
        # MQTT_DEBUG = True

    if config.get('mqtt:username') and config.get('mqtt:password'):
        MQTT_USER = config.get('mqtt:username')
        MQTT_PASSWD = config.get('mqtt:password')

    MQTT_HOST = config.get('mqtt:host', MQTT_HOST)
    MQTT_PORT = config.get('mqtt:port', MQTT_PORT)

    HOST = config.get('server:host', HOST)
    PORT = config.get('server:port', PORT)

    device_id = "epever"
    mqtt_topic = config.get('mqtt:topic', "epever/epever-solar")
    base_topic = mqtt_topic + '/{}'.format(device_id)

    publisher = mqtt.Client()
    epever_client = EpeverSolarClient(HOST, PORT)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    job = Job(interval=timedelta(seconds=WAIT_TIME_SECONDS),
              publisher=publisher, epever_client=epever_client)
    job.start()

    publisher.on_connect = on_connect
    publisher.on_message = on_message

    publisher.username_pw_set(username=MQTT_USER, password=MQTT_PASSWD)

    publisher.user_data_set(
        {"base_topic": base_topic, "epever_client": epever_client})
    publisher.connect(MQTT_HOST, MQTT_PORT, 60)

    publisher.loop_start()

    log_info("Running...")
    while True:
        try:
            time.sleep(1)
        except ProgramKilled:
            print("Program killed: running cleanup code")
            job.stop()
            break

    publisher.disconnect()
    publisher.loop_stop()


if __name__ == "__main__":
    main()
