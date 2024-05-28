package com.etfrogers.ecoforestklient

import kotlinx.datetime.LocalDate
import kotlinx.datetime.TimeZone
import kotlinx.datetime.format
import kotlinx.datetime.format.char
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.FormBody
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.security.SecureRandom
import java.security.cert.X509Certificate
import java.util.concurrent.TimeUnit
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager


const val CSV_DATE_FORMAT = "%Y/%m/%d %H:%M:%S"

class EcoForestClient(
    private val server: String,
    private val port: String,
    private val serialNumber: String,
    private val authKey: String,
    private val timezone: TimeZone? = null,
    debugSSL: Boolean = false
) {

    private var rawRegisterValues: MutableMap<Int, Int> = mutableMapOf()
    private val client: OkHttpClient

    init {
        var builder = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
        if (debugSSL) {
            builder = builder.ignoreAllSSLErrors()
        }
        client = builder.build()
    }

    internal fun apiRequest(endpoint: String, data: Map<String, Int>? = null /* **kwargs */): String {
        val url = "https://$server:$port/$endpoint"

//        ssl_verify = False  # os.path.abspath('easynet2.ecoforest.es.pem')
//        val standardArgs = mapOf(
//            "verify" to ssl_verify,
//            "headers" to mapOf ( "Authorization" to "Basic $authkey" )
//        )
        var requestBuilder = Request.Builder()
            .url(url)
            .addHeader("Authorization", "Basic $authKey")
        if (data != null) {
            var formBuilder = FormBody.Builder()
            for (item in data) {
                formBuilder = formBuilder.add(item.key, item.value.toString())
            }
            val formBody = formBuilder.build()
            requestBuilder = requestBuilder.post(formBody)
        }
        val request = requestBuilder.build()
        client.newCall(request).execute().use { response ->
            return response.body!!.string()
        }
    }

    private fun clearRegisterCache() {
        rawRegisterValues = mutableMapOf()
    }

    @OptIn(ExperimentalStdlibApi::class)
    internal fun readRegisterPage(page: RegisterPage) {
        val endpoint = "recepcion_datos_4.cgi"
        val action = if (page.type == RegisterType.BOOL) 2001 else 2002
        val postData = mapOf(
            "idOperacion" to action,
            "dir" to page.firstRegister,
            "num" to page.numberOfRegisters
        )
        val responseStr = apiRequest(endpoint, data = postData)
//        val responseStr = response.body!!.string()
        val tokens = responseStr.lines()
        val status = tokens[0]
        val payload = tokens[1]
        val (msg, errorCode) = status.split("=")
        val expectedMessage = if (page.type == RegisterType.BOOL) {
            "error_geo_get_bit"
        } else {
            "error_geo_get_reg"
        }
        if (msg != expectedMessage) {
            throw ServerCommsException("Error getting Ecoforest register (index: ${page.firstRegister}, " +
                    "length:${page.numberOfRegisters}, type: ${page.type}) " +
                    " - Expected message: ${expectedMessage}, Actual Message: {$msg}')")
        }
        if (errorCode != "0") {
            throw ServerCommsException(
                "Error getting Ecoforest register (index: ${page.firstRegister}, " +
                        "length:${page.numberOfRegisters}, type: ${page.type}) - Error code: ${errorCode}')"
            )
        }
        val payloadTokens = payload.split("&")
        val dir = payloadTokens[0]
        val num = payloadTokens[1]
        val binaryData = payloadTokens.drop(2)
        if (!(dir == "dir=${page.firstRegister}"
              && num == "num=${page.numberOfRegisters}"
              && binaryData.size == page.numberOfRegisters))
        {
            throw ServerCommsException(
                "Error getting Ecoforest data (index: $dir, expected ${page.firstRegister}, " +
                        "length:$num, reported: ${page.numberOfRegisters}, " +
                        "actual: ${binaryData.size})')"
            )
        }
        for (i in 0..<page.numberOfRegisters) {
            rawRegisterValues[page.firstRegister + i] = binaryData[i].hexToInt()
        }
    }

    private fun getRawRegisterValue(datapoint: StatusDatapoint): Int {
        var index = datapoint.registerIndex
        if (datapoint.type == RegisterType.INT) {
            index += 5001
        }
        if (!rawRegisterValues.containsKey(index)) {
            val page = findPage(datapoint.type, index)
            readRegisterPage(page)
        }
        return rawRegisterValues[index]!!
    }

    internal fun getIntRegisterValue(name: String): UnitValue<Int> {
        val datapoint = STATUS_DATAPOINTS[name]!!
        var value = getRawRegisterValue(datapoint)
        if (datapoint.signed && value > 32768) {
            value -= 65536
        }
        return UnitValue(value, datapoint.unit)
    }

    internal fun getFloatRegisterValue(name: String): UnitValue<Float> {
        val uValue = getIntRegisterValue(name)
        return UnitValue(uValue.value.toFloat() / 10, uValue.unit)
    }

    internal fun getBoolRegisterValue(name: String): Boolean {
        val uValue = getIntRegisterValue(name)
        return uValue.value != 0
    }

    private fun findPage(
        type: RegisterType,
        index: Int,
    ): RegisterPage {
        for (page in REGISTER_PAGES) {
            if (type == page.type
                && page.firstRegister <= index
                && index <= page.firstRegister + page.numberOfRegisters
            ) {
                return page
            }
        }
        throw RegisterPageNotFoundException("No register found for type $type and index $index")
    }

    fun getCurrentStatus(): EcoforestStatus {
        clearRegisterCache()
        return buildStatus(this)
    }
}

internal fun dateStr(date: LocalDate): String{
    return date.format(LocalDate.Format {
        year(); char('-'); monthNumber(); char('-'); dayOfMonth()
    })
}

internal enum class RegisterType {
    INT, FLOAT, BOOL
}
// Our name, EF Name, type, register index, 'unit', signed,
private data class StatusDatapoint(
    val name: String,
    val ecoforestName: String,
    val type: RegisterType,
    val registerIndex: Int,
    val unit: String,
    val signed: Boolean,
)

private val STATUS_DATAPOINTS = mapOf(
    "electricalPower" to StatusDatapoint("electricalPower", "e_elect", RegisterType.INT, 81, "W", true),
    "dhwActualTemp" to StatusDatapoint("dhwActualTemp", "temp_acum_acs", RegisterType.FLOAT, 8, "ºC", true),
    "dhwSetpoint" to StatusDatapoint("dhwSetpoint", "consigna_acs", RegisterType.FLOAT, 214, "ºC", true),
    "dhwOffset" to StatusDatapoint("dhwOffset", "offset", RegisterType.FLOAT, 15, "ºC", true),
    "outsideTemp" to StatusDatapoint("outsideTemp", "temp_exterior", RegisterType.FLOAT, 11, "ºC", true),
    "heatingBufferSetpoint" to StatusDatapoint("heatingBufferSetpoint", "set_inercia_heat", RegisterType.FLOAT, 215, "ºC", true),
    "heatingBufferActualTemp" to StatusDatapoint("heatingBufferActualTemp", "temp_dep_heat", RegisterType.FLOAT, 200, "ºC", true),
    "heatingBufferOffset" to StatusDatapoint("heatingBufferOffset", "offset_inercia_heat", RegisterType.FLOAT, 58, "ºC", true),
    "isHeatingOn" to StatusDatapoint("isHeatingOn", "top_1", RegisterType.BOOL, 206, "", false),
    "isHeatingDemand" to StatusDatapoint("isHeatingDemand", "top_1", RegisterType.BOOL, 249, "", false),
    "isDHWDemand" to StatusDatapoint("isDHWDemand", "acs", RegisterType.BOOL, 208, "", false),
)

data class UnitValue<T> (val value: T, val unit: String = "")
data class EcoforestStatus(
    val electricalPower: UnitValue<Int> = UnitValue(0),
    val dhwActualTemp: UnitValue<Float> = UnitValue(0f),
    val dhwSetpoint: UnitValue<Float> = UnitValue(0f),
    val dhwOffset: UnitValue<Float> = UnitValue(0f),
    val outsideTemp: UnitValue<Float> = UnitValue(0f),
    val heatingBufferActualTemp: UnitValue<Float> = UnitValue(0f),
    val heatingBufferSetpoint: UnitValue<Float> = UnitValue(0f),
    val heatingBufferOffset: UnitValue<Float> = UnitValue(0f),
    val isHeatingOn: Boolean = false,
    val isHeatingDemand: Boolean = false,
    val isDHWDemand: Boolean = false,
    )

internal fun buildStatus(client: EcoForestClient): EcoforestStatus {
    return EcoforestStatus(
        electricalPower = client.getIntRegisterValue("electricalPower"),
        dhwActualTemp = client.getFloatRegisterValue("dhwActualTemp"),
        dhwSetpoint = client.getFloatRegisterValue("dhwSetpoint"),
        dhwOffset = client.getFloatRegisterValue("dhwOffset"),
        outsideTemp = client.getFloatRegisterValue("outsideTemp"),
        heatingBufferActualTemp = client.getFloatRegisterValue("heatingBufferActualTemp"),
        heatingBufferSetpoint = client.getFloatRegisterValue("heatingBufferSetpoint"),
        heatingBufferOffset = client.getFloatRegisterValue("heatingBufferOffset"),
        isHeatingOn = client.getBoolRegisterValue("isHeatingOn"),
        isHeatingDemand = client.getBoolRegisterValue("isHeatingDemand"),
        isDHWDemand = client.getBoolRegisterValue("isDHWDemand"),
    )
}


class RegisterPageNotFoundException(message:String): Exception(message)
class ServerCommsException(message:String): Exception(message)

internal data class RegisterPage(
    val firstRegister: Int,
    val numberOfRegisters: Int,
    val type: RegisterType,
)

// pages of registers used, as extracted from informacion.js
private val REGISTER_PAGES = listOf(
    RegisterPage(61, 25, RegisterType.BOOL),
    RegisterPage(101, 97, RegisterType.BOOL),
    RegisterPage(206, 62, RegisterType.BOOL),
    RegisterPage(5033, 2, RegisterType.INT),
    RegisterPage(5066, 18, RegisterType.INT),

    RegisterPage(5113, 31, RegisterType.INT),
    RegisterPage(5185, 27, RegisterType.INT),
    RegisterPage(5241, 34, RegisterType.INT),
    RegisterPage(5285, 14, RegisterType.INT),
    RegisterPage(1, 39, RegisterType.FLOAT),

    RegisterPage(40, 19, RegisterType.FLOAT),
    RegisterPage(97, 30, RegisterType.FLOAT),
    RegisterPage(176, 29, RegisterType.FLOAT),
    RegisterPage(214, 14, RegisterType.FLOAT)
)

@Serializable
data class EcoForestConfig(
    val server: String,
    val port: String,
    @SerialName("serial-number") val serialNumber: String,
    @SerialName("auth-key") val authKey: String,
)

fun OkHttpClient.Builder.ignoreAllSSLErrors(): OkHttpClient.Builder {
    val naiveTrustManager = object : X509TrustManager {
        override fun getAcceptedIssuers(): Array<X509Certificate> = arrayOf()
        override fun checkClientTrusted(certs: Array<X509Certificate>, authType: String) = Unit
        override fun checkServerTrusted(certs: Array<X509Certificate>, authType: String) = Unit
    }

    val insecureSocketFactory = SSLContext.getInstance("TLSv1.2").apply {
        val trustAllCerts = arrayOf<TrustManager>(naiveTrustManager)
        init(null, trustAllCerts, SecureRandom())
    }.socketFactory

    sslSocketFactory(insecureSocketFactory, naiveTrustManager)
    hostnameVerifier(HostnameVerifier { _, _ -> true })
    return this
}

fun main(){
    val text = File("config.json").readText()
    val config = Json.decodeFromString<EcoForestConfig>(text)
    val client = EcoForestClient(
        server = config.server,
        port = config.port,
        serialNumber = config.serialNumber,
        authKey = config.authKey,
        debugSSL = true,
    )

    val endpoint = "recepcion_datos_4.cgi"
    val action = /*2001 if type_ is bool else*/ 2002
    val page = RegisterPage(5033, 3, RegisterType.INT)
    val data = mapOf("idOperacion" to action, "dir" to page.firstRegister, "num" to page.numberOfRegisters)
//    val response = client.apiRequest(endpoint, data)
//    val response = client.readRegisterPage(REGISTER_PAGES[3])
    val response = client.getCurrentStatus()
    println(response)
//    println(response.body!!.string())
}

/*
import calendar
import datetime
import pathlib
import warnings
from typing import Tuple

import requests

import numpy as np
import yaml
from urllib3.exceptions import InsecureRequestWarning

from ecoforest.history_dataset import DayData, CompositeDataSet, MonthDataSet

CSV_DATE_FORMAT = '%Y/%m/%d %H:%M:%S'


class EcoforestClient:


    def get_history_for_date_range(self, dates: Tuple[datetime.date, datetime.date]):
        return CompositeDataSet([self.get_history_for_date(date) for date in date_range(*dates)])

    def get_history_for_month(self, year: int, month: int):
        _, n_days = calendar.monthrange(year, month)
        composite = self.get_history_for_date_range((datetime.date(year, month, 1),
                                                     datetime.date(year, month, n_days)))
        return MonthDataSet(composite.datasets)

    def get_history_for_date(self, date: datetime.date):
        use_cache = True
        if use_cache:
            cache_file = self._history_cache_file(date)
            try:
                with open(cache_file) as file:
                    contents = file.read()
            except FileNotFoundError:
                contents = self.get_history_data_from_server(date)
                if contents and date != datetime.datetime.today().date():
                    # do not cache today's data as it will change
                    with open(cache_file, 'w') as file:
                        file.write(contents)
        else:
            contents = self.get_history_data_from_server(date)
        timestamps, full_data = self.process_file_data(contents, self.timezone)
        return DayData(timestamps, full_data)

    @staticmethod
    def _history_cache_file(date: datetime.date):
        data_dir = pathlib.Path('cache/ecoforest')
        return data_dir / f'{EcoforestClient.date_str(date)}.csv'


    def get_history_data_from_server(self, date: datetime.date):
        data_dir = 'historic'
        filename = f'{self.date_str(date)}_{self.serial_number}_1_historico.csv'
        response = self.api_request(f'{data_dir}/{filename}')
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            if response.status_code == 404:
                return ''
            else:
                raise
        return response.text

    @staticmethod
    def process_file_data(contents, timezone=None):
        headers, *lines = contents.split('\n')
        timestamps = []
        data = []
        for line in lines:
            if line:
                _, timestamp, *entry = line.split(';')[:-1]
                time = datetime.datetime.strptime(timestamp, CSV_DATE_FORMAT)
                time = time.replace(tzinfo=timezone)
                # time = time.replace(tzinfo=zoneinfo.ZoneInfo('UTC'))
                # time = time.astimezone(timezone)
                timestamps.append(time)
                data.append([float(val) for val in entry])
        if data:
            data = np.array(data) / 10
        else:
            data = np.zeros((0, 30))  # each file has 30 columns, here we set n_rows to 0
        return np.array(timestamps), data

    def _get_register_values(self, first_register: int, number_of_registers: int, type_: type):
        endpoint = 'recepcion_datos_4.cgi'
        action = 2001 if type_ is bool else 2002
        data = {"idOperacion": action, 'dir': first_register, 'num': number_of_registers}
        response = self.api_request(endpoint, data=data)
        data = response.text
        status, data, *null = data.split("\n")
        msg, error_code = status.split("=")
        if type_ is bool:
            assert msg == 'error_geo_get_bit'
        else:
            assert msg == 'error_geo_get_reg'
        if error_code != '0':
            raise RuntimeError(f'Error getting Ecoforest register (index: {first_register}, '
                               f'length:{number_of_registers}, type: {type_}) - Error code: {error_code}')
        dir, num, *binary_data = data.split("&")
        assert dir == f'dir={first_register}'
        assert num == f'num={number_of_registers}'
        assert len(binary_data) == number_of_registers
        for i in range(number_of_registers):
            self._register_values[first_register + i] = binary_data[i]

    def _register_value(self, index, type_):
        if index not in self._register_values:
            for page in REGISTER_PAGES:
                first_register, number_of_registers, page_type = page
                if (type_ is page_type
                        and first_register <= index <= first_register + number_of_registers):
                    break
            else:
                first_register = number_of_registers = None
            assert first_register is not None
            self._get_register_values(first_register, number_of_registers, type_)
        return self._register_values[index]

    def _clear_register_cache(self):
        self._register_values = {}

    def get_current_status(self):
        self._clear_register_cache()
        status = {}
        for line in STATUS_DATAPOINTS:
            name, _, type_, index, unit, signed, *_ = line
            if type_ is int:
                index += 5001
            val = int(self._register_value(index, type_), base=16)
            if signed and val > 32768:
                val -= 65536
            if type_ is float:
                val /= 10
            elif type_ is bool:
                val = bool(val)
            status[name] = {'value': val, 'unit': unit}
        return status


# Our name, EF Name, type, register index, 'unit', signed,
STATUS_DATAPOINTS = [['ElectricalPower', 'e_elect', int, 81, 'W', True],
                     ['DHWActualTemp', 'temp_acum_acs', float, 8, 'ºC', True],
                     ['DWHSetpoint', 'consigna_acs', float, 214, 'ºC', True],
                     ['OutsideTemp', 'temp_exterior', float, 11, 'ºC', True],
                     ['HeatingBufferSetpoint', 'set_inercia_heat', float, 215, 'ºC', True],
                     ['HeatingBufferActualTemp', 'temp_dep_heat', float, 200, 'ºC', True],
                     ['HeatingBufferOffset', 'offset_inercia_heat', float, 58, 'ºC', True],
                     ['HeatingOn', 'top_1', bool, 206, '', False],
                     ['HeatingDemand', 'top_1', bool, 249, '', False],
                     ['DHWDemand', 'acs', bool, 208, '', False],
                     ]

# pages of registers used, as extracted from informacion.js
REGISTER_PAGES = [(61, 25, bool), (101, 97, bool,), (206, 62, bool), (5033, 2, int), (5066, 18, int),
                  (5113, 31, int), (5185, 27, int), (5241, 34, int), (5285, 14, int), (1, 39, float),
                  (40, 19, float), (97, 30, float), (176, 29, float), (214, 14, float)]


def date_range(start_date: datetime.date, end_date: datetime.date):
    date = start_date
    while date <= end_date:
        yield date
        date += datetime.timedelta(days=1)


def main():
    year = 2023
    month = 3
    # day = 10
    # datasets = []
    # for day in range(15, 21):
    #     data = DayData(datetime.date(year, month, day))
    #     data.plot()
    #     datasets.append(data)
    with open('site_config.yml') as file:
        config = yaml.safe_load(file)
    config = config['ecoforest']
    client = EcoforestClient(config['server'], config['port'], config['serial-number'], config['auth-key'])
    # client.get_current_status()

    dataset = client.get_history_for_month(year, month)
    # print(dataset.mean_cop())
    # print(dataset.mean_cop(ChunkClass.DHW))
    # print([c.type for c in dataset.chunks()])
    # print([d.mean_cop() for d in dataset.datasets])
    dataset.plot_bar_chart()


if __name__ == '__main__':
    main()

 */