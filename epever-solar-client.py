"""Epever solar client"""

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

import paho.mqtt.publish as mqtt

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


def value32(low, high):
    """value32"""
    return ctypes.c_int(low + (high << 16)).value / 100


def value16(value):
    """value16"""
    return ctypes.c_short(value).value / 100


def value8(value):
    """value8"""
    return [value >> 8, value & 0xFF]


def to_bool(value):
    """to_bool"""
    values = {
        0: False,
        1: True
    }
    return values.get(value, None)


def days(value):
    """days"""
    return f"Days {value}"


def minutes(value):
    """minutes"""
    return f"Minutes {value}"


def seconds(value):
    """seconds"""
    return f"Seconds {value}"


def hour_minute(value):
    """hour_minute"""
    hm = value8(value)
    return f"{hm[0]} hours {hm[1]} minutes"


def get_time(second, minute, hour):
    """get_time"""
    return datetime.time(hour, minute, second)


def to_float(low, high=None):
    """to_float"""
    if high is None:
        return value16(low)
    else:
        return value32(low, high)


class MQTTPublisher:
    """MQTT publisher"""
    mqtt_host = '127.0.0.1'
    mqtt_port = 1883

    mqtt_user = ''
    mqtt_pass = ''
    mqtt_topic = ''

    def __init__(self, host, port, user, passwd, topic):
        self.mqtt_host = host
        self.mqtt_port = port
        self.mqtt_user = user
        self.mqtt_pass = passwd
        self.mqtt_topic = topic

    def publish(self, device_id: str, event_name: str, event: object):
        """Publish event"""
        auth_data = None
        if (self.mqtt_user and self.mqtt_pass):
            auth_data = {'username': self.mqtt_user,
                         'password': self.mqtt_pass}

        payload = json.dumps(event)
        topic = self.mqtt_topic + f"/{device_id}/{event_name}"
        mqtt.single(topic=topic, payload=payload, retain=False,
                    hostname=self.mqtt_host, port=self.mqtt_port, auth=auth_data, tls=None)


class EpeverSolarClient:
    """EpeverSolarClient"""
    _host = "192.168.88.78"
    _port = 9999

    def __init__(self, host, port):
        self._host = host
        self._port = port

    def get_device_info(self, client):
        """get_device"""
        request = ReadDeviceInformationRequest(slave=1)
        response = client.execute(request)
        if not response.isError():
            result = []
            for idx in response.information:
                result.append(response.information[idx].decode("ascii"))
            return result

        logger.error(response)
        return None

    def init_clock(self, client):
        """init_clock"""
        print("Updating Device RTC...")
        result = client.read_holding_registers(0x9013, 3, slave=1)
        if not result.isError():
            logger.info("Date(%s, %s, %s, %s, %s, %s)",
                        2000 + (result.registers[2] >> 8),
                        result.registers[2] & 0xFF,
                        result.registers[1] >> 8,
                        result.registers[1] & 0xFF,
                        result.registers[0] >> 8,
                        result.registers[0] & 0xFF)

            now = datetime.datetime.now()
            new_data = [0, 0, 0]
            new_data[2] = ((now.year - 2000) << 8) + now.month
            new_data[1] = (now.day << 8) + now.hour
            new_data[0] = (now.minute << 8) + now.second
            logger.info("Date(%s, %s, %s, %s, %s, %s)",
                        2000 + (new_data[2] >> 8),
                        new_data[2] & 0xFF,
                        new_data[1] >> 8,
                        new_data[1] & 0xFF,
                        new_data[0] >> 8,
                        new_data[0] & 0xFF)

            result = client.write_registers(0x9013, new_data, slave=1)
            if not result.isError():
                print("Err:", "Updating Device RTC")
            else:
                print("Device RTC Updated.")

    def init_settings(self, client, settings):
        """init_settings"""
        print("Configuring battery setting...")
        result = client.write_registers(0x9000, settings, slave=1)
        if not result.isError():
            print("Battery setting done.")
        else:
            print("Err:", "Battery setting")

    def configure_battery(self, client):
        """configure_battery"""
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
        """get_battery_settings"""
        settings = {}
        result = client.read_holding_registers(0x9000, 15, slave=1)
        if not result.isError():
            battery_type = {
                0: "User defined",
                1: "Sealed",
                2: "GEL",
                3: "Flooded"
            }
            settings["batteryType"] = battery_type.get(result.registers[0])
            settings["batteryCapacity"] = result.registers[1]
            settings["temperatureCompensationCoefficient"] = to_float(
                result.registers[2])
            settings["highVoltDisconnect"] = to_float(result.registers[3])
            settings["chargingLimitVoltage"] = to_float(result.registers[4])
            settings["overVoltageReconnect"] = to_float(result.registers[5])
            settings["equalizationVoltage"] = to_float(result.registers[6])
            settings["boostVoltage"] = to_float(result.registers[7])
            settings["floatVoltage"] = to_float(result.registers[8])
            settings["boostReconnectVoltage"] = to_float(result.registers[9])
            settings["lowVoltageReconnect"] = to_float(result.registers[10])
            settings["underVoltageRecover"] = to_float(result.registers[11])
            settings["underVoltageWarning"] = to_float(result.registers[12])
            settings["lowVoltageDisconnect"] = to_float(result.registers[13])
            settings["dischargingLimitVoltage"] = to_float(
                result.registers[14])
        else:
            logger.error(result)
            return None

        return settings

    def get_battery_load(self, client):
        """get_battery_load"""
        result = client.read_input_registers(0x310C, 4, slave=1)
        if not result.isError():
            all_data = {}
            all_data["loadVoltage"] = to_float(result.registers[0])
            all_data["loadCurrent"] = to_float(result.registers[1])
            all_data["loadPower"] = to_float(
                result.registers[2], result.registers[3])

            return all_data
        else:
            logger.error(result)
            return None

    def get_data(self, client):
        """get_data"""
        data = {}

        result = client.read_input_registers(0x3100, 19, slave=1)
        if not result.isError():
            data["chargingInputVoltage"] = to_float(result.registers[0])
            data["chargingInputCurrent"] = to_float(result.registers[1])
            data["chargingInputPower"] = to_float(
                result.registers[2], result.registers[3])

            data["chargingOutputVoltage"] = to_float(result.registers[4])
            data["chargingOutputCurrent"] = to_float(result.registers[5])
            data["chargingOutputPower"] = to_float(
                result.registers[6], result.registers[7])

            data["dischargingOutputVoltage"] = to_float(result.registers[12])
            data["dischargingOutputVurrent"] = to_float(result.registers[13])
            data["dischargingOutputPower"] = to_float(
                result.registers[14], result.registers[15])

            data["batteryTemperature"] = to_float(result.registers[16])
            data["temperatureInside"] = to_float(result.registers[17])
            data["powerComponentsTemperature"] = to_float(result.registers[18])
        else:
            logger.error(result)
            return None

        result = client.read_input_registers(0x311A, 2, slave=1)
        if not result.isError():
            data["batterySoC"] = to_float(result.registers[0]) * 100
            data["remoteBatteryTemperature"] = to_float(result.registers[1])
        else:
            logger.error(result)
            return None

        result = client.read_input_registers(0x311D, 1, slave=1)
        if not result.isError():
            data["batteryRealRatedPower"] = to_float(result.registers[0])
        else:
            logger.error(result)
            return None

        return data

    def get_battery_stat(self, client):
        """get_battery_stat"""
        result = client.read_input_registers(0x3300, 31, slave=1)
        if not result.isError():
            stat = {}
            stat["maxVoltToday"] = to_float(result.registers[0])
            stat["minVoltToday"] = to_float(result.registers[1])
            stat["maxBatteryVoltToday"] = to_float(result.registers[2])
            stat["minBatteryVoltToday"] = to_float(result.registers[3])
            stat["consumedEnergyToday"] = to_float(
                result.registers[4], result.registers[5])
            stat["consumedEnergyMonth"] = to_float(
                result.registers[6], result.registers[7])
            stat["consumedEnergyYear"] = to_float(
                result.registers[8], result.registers[9])
            stat["totalConsumedEnergy"] = to_float(
                result.registers[10], result.registers[11])
            stat["generatedEnergyToday"] = to_float(
                result.registers[12], result.registers[13])
            stat["generatedEnergyMonth"] = to_float(
                result.registers[14], result.registers[15])
            stat["generatedEnergyYear"] = to_float(
                result.registers[16], result.registers[17])
            stat["totalGeneratedEnergy"] = to_float(
                result.registers[18], result.registers[19])
            stat["carbonDioxideReduction"] = to_float(
                result.registers[20], result.registers[21])
            stat["batteryVoltage"] = to_float(result.registers[26])
            stat["batteryCurrent"] = to_float(
                result.registers[27], result.registers[28])
            stat["batteryTemperature"] = to_float(result.registers[29])
            stat["ambientTemperature"] = to_float(result.registers[30])
            return stat
        else:
            logger.error(result)
            return None

    def get_battery_status(self, client):
        """get_battery_status"""
        result = client.read_input_registers(0x3200, 3, slave=1)
        if not result.isError():
            value = result.registers[0]
            battery_status_voltage = {
                0: "Normal",
                1: "Overvolt",
                2: "Under volt",
                3: "Low Volt Disconnect",
                4: "Fault"
            }
            battery_status_temperature = {
                0: "Normal",
                1: "Over Temperature",
                2: "Low Temperature"
            }
            abnormal_status = {
                0: "Normal",
                1: "Abnormal"
            }
            wrong_status = {
                0: "Correct",
                1: "Wrong"
            }
            battery_status = {}
            battery_status["voltage"] = battery_status_voltage.get(
                value & 0x0007, None)
            battery_status["temperature"] = battery_status_temperature.get(
                (value >> 4) & 0x000f, None)
            battery_status["internalResistance"] = abnormal_status.get(
                (value >> 8) & 0x0001, None)
            battery_status["ratedVoltage"] = wrong_status.get(
                (value >> 15) & 0x0001, None)

            value = result.registers[1]
            charging_equipment_status_input_voltage = {
                0: "Normal",
                1: "No power connected",
                2: "Higher Volt Input",
                3: "Input Volt Error"
            }
            charging_equipment_status_battery = {
                0: "Not charging",
                1: "Float",
                2: "Boost",
                3: "Equalization"
            }
            fault_status = {
                0: "Normal",
                1: "Fault"
            }
            running_status = {
                0: "Standby",
                1: "Running"
            }

            equipment_status = {}
            equipment_status["inputVoltage"] = charging_equipment_status_input_voltage.get(
                (value >> 14) & 0x0003, None)
            equipment_status["mosfetShort"] = to_bool((value >> 13) & 0x0001)
            equipment_status["chargingAntiReverseMosfetShort"] = to_bool(
                (value >> 12) & 0x0001)
            equipment_status["antiReverseMosfetShort"] = to_bool(
                (value >> 11) & 0x0001)
            equipment_status["inputOverCurrent"] = to_bool(
                (value >> 10) & 0x0001)
            equipment_status["loadOverCurrent"] = to_bool(
                (value >> 9) & 0x0001)
            equipment_status["loadShort"] = to_bool((value >> 8) & 0x0001)
            equipment_status["loadMosfetShort"] = to_bool(
                (value >> 7) & 0x0001)
            equipment_status["pvInputShort"] = to_bool((value >> 4) & 0x0001)
            equipment_status["battery"] = charging_equipment_status_battery.get(
                (value >> 2) & 0x0003, None)
            equipment_status["fault"] = fault_status.get(
                (value >> 1) & 0x0001, None)
            equipment_status["running"] = running_status.get(
                (value) & 0x0001, None)

            value = result.registers[2]
            discharging_equipment_status_voltage = {
                0: "Normal",
                1: "Low",
                2: "High",
                3: "No access input volt error"
            }
            discharging_equipment_status_output = {
                0: "Light Load",
                1: "Moderate",
                2: "Rated",
                3: "Overload"
            }
            discharging_equipment_status = {}

            discharging_equipment_status["inputVoltage"] = discharging_equipment_status_voltage.get(
                (value >> 14) & 0x0003, None)
            discharging_equipment_status["outputPower"] = discharging_equipment_status_output.get(
                (value >> 12) & 0x0003, None)
            discharging_equipment_status["shortCircuit"] = to_bool(
                (value >> 11) & 0x0001)
            discharging_equipment_status["unableDischarge"] = to_bool(
                (value >> 10) & 0x0001)
            discharging_equipment_status["unableStopDischarging"] = to_bool(
                (value >> 9) & 0x0001)
            discharging_equipment_status["outputVoltageAbnormal"] = to_bool(
                (value >> 8) & 0x0001)
            discharging_equipment_status["inputOverpressure"] = to_bool(
                (value >> 7) & 0x0001)
            discharging_equipment_status["highVoltageSideShortCircuit"] = to_bool(
                (value >> 6) & 0x0001)
            discharging_equipment_status["boostOverpressure"] = to_bool(
                (value >> 5) & 0x0001)
            discharging_equipment_status["outputOverpressure"] = to_bool(
                (value >> 4) & 0x0001)

            discharging_equipment_status["fault"] = fault_status.get(
                (value >> 1) & 0x0001, None)
            discharging_equipment_status["running"] = running_status.get(
                value & 0x0001, None)

            return {"batteryStatus": battery_status, "equipmentStatus": equipment_status,
                    "dischargingEquipmentStatus": discharging_equipment_status}
        else:
            logger.error(result)
            return None

    def run(self, method='get_data'):
        """run"""
        client = ModbusTcpClient(
            self._host, port=self._port, framer=ModbusFramer, retries=5)

        logger.info('Running method: %s', method)
        try:
            success = client.connect()
            if not success:
                logger.error("Cannot connect to Epever server")
                return None

            client.send(bytes.fromhex("20020000"))
            run_method = getattr(EpeverSolarClient, method)
            return run_method(self, client)

        except Exception as e:
            logger.error(e)

        finally:
            client.close()


class Configurator:
    """Configurator"""
    config = {}

    def __init__(self, config):
        """init"""
        self.config = config

    def get(self, path, default=None):
        """get"""
        items = path.split(':')
        value = None

        for idx, item in enumerate(items):
            if idx == 0:
                value = self.config.get(item)
            else:
                value = None if value is None else value.get(item)

        return value or default


class ProgramKilled(Exception):
    """ProgramKilled"""
    logger.info("Program terminated.")


class Job(threading.Thread):
    """Job"""

    def __init__(self, interval, *args, **kwargs):
        """init"""
        threading.Thread.__init__(self)
        self.daemon = False
        self.stopped = threading.Event()
        self.interval = interval
        self.args = args
        self.kwargs = kwargs
        self.init = True

    def execute(self, publisher: MQTTPublisher, epever_client: EpeverSolarClient,
                device_id: str):
        """execute"""
        if self.init:
            data = epever_client.run("get_device_info")
            if data is not None:
                publisher.publish(device_id, "info", data)
            else:
                logger.error("get_device_info")

            data = epever_client.run("get_battery_settings")
            if data is not None:
                publisher.publish(device_id, "settings", data)
            else:
                logger.error("get_battery_settings")

        data = epever_client.run("get_data")
        if data is not None:
            publisher.publish(device_id, "data", data)
        else:
            logger.error("get_data")

        if datetime.datetime.now().minute % 5 == 0 or self.init:
            data = epever_client.run("get_battery_stat")
            if data is not None:
                publisher.publish(device_id, "stat", data)
            else:
                logger.error("get_battery_stat")

            data = epever_client.run("get_battery_status")
            if data is not None:
                publisher.publish(device_id, "status", data)
            else:
                logger.error("get_battery_status")

            data = epever_client.run("get_battery_load")
            if data is not None:
                publisher.publish(device_id, "load", data)
            else:
                logger.error("get_battery_load")

        if self.init:
            self.init = False

    def stop(self):
        """stop"""
        self.stopped.set()
        self.join()

    def run(self):
        """run"""
        self.execute(*self.args, **self.kwargs)
        # looping
        while not self.stopped.wait(self.interval.total_seconds()):
            self.execute(*self.args, **self.kwargs)


WAIT_TIME_SECONDS = 60


def signal_handler(signum, frame):
    """signal_handler"""
    raise ProgramKilled


def main():
    """main"""
    logger.info("Epever solar client started...")

    mqtt_host = "localhost"
    mqtt_port = 1883
    mqtt_user = ""
    mqtt_pass = ""

    config = None
    if os.path.exists('./epever-solar-client.json'):
        logger.info('Running in local mode')
        with open('./epever-solar-client.json', 'r', encoding="utf-8") as fp:
            config = Configurator(json.load(fp))
            fp.close()
    elif os.path.exists('/data/options.json'):
        logger.info('Running in hass.io add-on mode')
        with open('/data/options.json', 'r', encoding="utf-8") as fp:
            config = Configurator(json.load(fp))
            fp.close()
    else:
        logger.info('Configuration file not found, exiting.')
        sys.exit(1)

    if config.get('mqtt:debug'):
        logger.info("Debugging messages enabled")
        # MQTT_DEBUG = True

    if config.get('mqtt:username') and config.get('mqtt:password'):
        mqtt_user = config.get('mqtt:username')
        mqtt_pass = config.get('mqtt:password')

    mqtt_host = config.get('mqtt:host', mqtt_host)
    mqtt_port = config.get('mqtt:port', mqtt_port)

    mqtt_topic = config.get('mqtt:topic', "epever/epever-solar")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    publisher = MQTTPublisher(mqtt_host, mqtt_port,
                              mqtt_user, mqtt_pass, mqtt_topic)

    servers = config.get("server")
    for server in servers:
        host, port = '0.0.0.0', 15002
        device_id = "epever"

        device_id = server.get('name', device_id)
        host = server.get('host', host)
        port = server.get('port', port)

        epever_client = EpeverSolarClient(host, port)

        job = Job(interval=timedelta(seconds=WAIT_TIME_SECONDS),
                  publisher=publisher,
                  epever_client=epever_client,
                  device_id=device_id)
        job.start()

    logger.info("Running...")
    while True:
        try:
            time.sleep(1)
        except ProgramKilled:
            print("Program killed: running cleanup code")
            job.stop()
            break


if __name__ == "__main__":
    main()
