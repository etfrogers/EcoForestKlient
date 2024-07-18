package com.etfrogers.ecoforestklient

import kotlinx.datetime.LocalDate
import kotlinx.datetime.LocalDateTime
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toInstant
import kotlin.time.Duration

private fun  List<Int>.diff(): List<Int> {
    return this.drop(1).zip(this.dropLast(1)).map {(second, first) -> second-first}
}

private fun List<Number>.weightedMean(weights: List<Number>): Double {
    return this.zip(weights).map { (v, w) -> v.toFloat() * w.toFloat() }.average()
}

inline fun <E> Iterable<E>.indexesOf(predicate: (E) -> Boolean)
        = mapIndexedNotNull{ index, elem -> index.takeIf{ predicate(elem) } }

private const val TANK_OFFSET_TEMP = 5
private const val DHW_OFFSET_TEMP = 4
private const val DHW_SOLAR_SETPOINT = 55
private const val DHW_LEGIONNAIRES_SETPOINT = 65
private const val HEATING_SOLAR_SETPOINT = 60
private const val DHW_SETPOINT = 48
private const val TEMPERATURE_TOLERANCE = 2

private const val EXPLORE_UNUSED_DATA = false


private operator fun List<Number>.div(other: List<Number>): List<Float> {
    return this.zip(other).map { (t, o) -> t.toFloat() / o.toFloat()}
}

abstract class BaseDataset(
    private val timezone: TimeZone
) {
    // indices taken from Javascript plotting code
    // (need to subtract 2 from the indices used there to allow for serial and timestamp)
    val MAPPING = mapOf(
        "consumption" to 24,
        "heating" to 25,
        "cooling" to 26,
        "dhwTemp" to 10,
        // no index for "pool"
        "heatingBufferTemp" to 13,
        "coolingBufferTemp" to 14,
        "productionSupply" to 15,
        "productionReturn" to 16,
        "brineSupply" to 17,
        "brineReturn" to 18,
        "outdoorTemp" to 11,
        "zone1" to 19,
        "zone2" to 20,
        "zone3" to 21,
        "zone4" to 22,
    )


    abstract val timestamps: List<LocalDateTime>
    abstract val data: List<List<Float>>
    abstract val heating: List<Float>
    abstract val consumption: List<Float>
    abstract val cooling: List<Float>
    abstract val dhwTemp: List<Float>
    abstract val heatingBufferTemp: List<Float>
    abstract val coolingBufferTemp: List<Float>
    abstract val productionSupply: List<Float>
    abstract val productionReturn: List<Float>
    abstract val brineSupply: List<Float>
    abstract val brineReturn: List<Float>
    abstract val outdoorTemp: List<Float>
    val size: Int
        get() = timestamps.size


    val length: Duration
        get() =  timestamps.last().toInstant(timezone) - timestamps[0].toInstant(timezone)

    companion object {
        fun totalPower(series: List<Float>): Float {
            // 5 minute intervals, means 12 to an hour, so one 1kW at one point is 1/12 kWh
            return series.sum() / 12
        }
    }

    fun heatingEnergyOfType(types: Set<ChunkClass>): Float {
        return chunks(types).map { it.totalHeating }.sum()
//        return sum([c.total_heating for c in chunks(types)])
    }
    fun consumedEnergyOfType(types: Set<ChunkClass>): Float {
        return chunks(types).map { it.totalConsumption }.sum()
//        return sum([c.total_consumption for c in self.chunks(types)])
    }

    val totalConsumption: Float
        get() = totalPower(consumption)


    val totalHeating: Float
        get() = totalPower(heating)


    fun instantaneousCOP(): List<Float> {
        return (heating / consumption).map { if (it.isNaN()) 0f else it }
    }

    fun cop(): Double {
        return instantaneousCOP().average()
    }

    fun cop(types: Set<ChunkClass>): Double  {
        if (totalHeating == 0f) {
            return 0.0
        }

        val chunks = chunks(types)
        if (chunks.isNotEmpty()) {
            val weights = chunks.map { it.length.inWholeMinutes }
            return chunks.map { it.cop() }.weightedMean(weights)
        } else {
            return 0.0
        }
    }

    fun chunks(type: ChunkClass): List<DataChunk>{
        return chunks(setOf(type))
    }

    //cache?
    fun chunks(types: Set<ChunkClass>? = null): List<DataChunk> {
       return if (types != null) {
            chunkList.filter { it.type in types }
        } else {
            chunkList
        }
    }

    private val chunkList: List<DataChunk>  by lazy {
        chunkGenerator().asSequence().toList()
    }

    private fun chunkGenerator(): Iterator<DataChunk> {
        return iterator {
            val isOn = consumption.map { if(it != 0f) 1 else 0 }
            val switches = isOn.diff()
            val starts = (switches.indexesOf { it == 1 }).toMutableList()
            val ends = (switches.indexesOf { it == -1 }).toMutableList()
            if (isOn[0] == 1) {
                starts.add(0, 0)
            }
            if (isOn.last()==1) {
                ends.add(isOn.size)
            }
            assert(starts.size == ends.size)
            // returns inclusive inds (first non element and last nonzero element)
            starts.map { it+1 }.zip(ends).forEach {(startInd, endInd) ->
                yield(DataChunk(
                    timezone,
                    timestamps.slice(startInd..endInd),
                    data.map { it.slice(startInd..endInd) } )
                )
            }
//            for (start_ind, end_ind in zip(starts+1, ends):
//                yield DataChunk (self.timestamps[start_ind:end_ind+1], self.data[start_ind:end_ind+1])
        }
    }
    fun isEmpty() = timestamps.isEmpty()

    fun getPowerSeries(types: Set<ChunkClass>): List<Float> {
        val power = Array(timestamps.size) { 0f }
        for (chunk in chunks(types)) {
            val startInd = timestamps.indexOf(chunk.timestamps[0])
            for (i in startInd..<startInd+chunk.size) {
                power[i] += chunk.consumption[i-startInd]
            }
        }
        return power.toList()
    }
}

open class Dataset(
    timezone: TimeZone,
    override val timestamps: List<LocalDateTime>,
    final override val data: List<List<Float>>,
): BaseDataset(timezone) {
    override val cooling = data[MAPPING["cooling"]!!]
    override val heating = data[MAPPING["heating"]!!]
    override val consumption = data[MAPPING["consumption"]!!]
    override val dhwTemp = data[MAPPING["dhwTemp"]!!]
    override val heatingBufferTemp = data[MAPPING["heatingBufferTemp"]!!]
    override val coolingBufferTemp = data[MAPPING["coolingBufferTemp"]!!]
    override val productionSupply = data[MAPPING["productionSupply"]!!]
    override val productionReturn = data[MAPPING["productionReturn"]!!]
    override val brineSupply = data[MAPPING["brineSupply"]!!]
    override val brineReturn = data[MAPPING["brineReturn"]!!]
    override val outdoorTemp = data[MAPPING["outdoorTemp"]!!]
    }

enum class ChunkClass {
    DHW,
    HEATING,
    COMBINED,
    SOLAR_DHW,
    SOLAR_HEATING,
    SOLAR_COMBINED,
    LEGIONNAIRES,
    LEGIONNAIRES_COMBINED,
    UNKNOWN;

    companion object {
        fun solarTypes(): Set<ChunkClass> = setOf(SOLAR_HEATING, SOLAR_DHW, SOLAR_COMBINED)
        fun heatingTypes(): Set<ChunkClass> = setOf(SOLAR_HEATING, HEATING)
        fun dhwTypes(): Set<ChunkClass> = setOf(DHW, SOLAR_DHW)
        fun combinedTypes(): Set<ChunkClass> =
            setOf(COMBINED, SOLAR_COMBINED, LEGIONNAIRES_COMBINED)
        fun legionnairesTypes(): Set<ChunkClass> = setOf(LEGIONNAIRES, LEGIONNAIRES_COMBINED)
    }
}

class ChunkTypeError: Exception()


class DataChunk(
    timezone: TimeZone,
    timestamps: List<LocalDateTime>,
    data: List<List<Float>>
) : Dataset(timezone, timestamps, data) {

    val type: ChunkClass
        get() {
            var dhwSolar = false
            var dwhLegionnaires = false
            var heatingSolar = false
            val endDhwTemp = dhwTemp.last()
            val dhwDiff = endDhwTemp - dhwTemp[0]
            val dhwIncreased = dhwDiff > DHW_OFFSET_TEMP - TEMPERATURE_TOLERANCE
            if (dhwIncreased) {
//                println("End temp: $endDhwTemp")
                dhwSolar = DHW_SOLAR_SETPOINT < endDhwTemp && endDhwTemp < DHW_LEGIONNAIRES_SETPOINT
                dwhLegionnaires =
                    endDhwTemp > DHW_LEGIONNAIRES_SETPOINT - TEMPERATURE_TOLERANCE
            }
            val endHeatingTemp = heatingBufferTemp.last()
            val heatingDiff = endHeatingTemp - heatingBufferTemp[0]
            val heatingIncreased = heatingDiff > TANK_OFFSET_TEMP - TEMPERATURE_TOLERANCE
            if (heatingIncreased) {
                heatingSolar = endHeatingTemp > HEATING_SOLAR_SETPOINT - TEMPERATURE_TOLERANCE
            }

            when {
                dwhLegionnaires -> {
                    return if (heatingIncreased) {
                        ChunkClass.LEGIONNAIRES_COMBINED
                    } else {
                        ChunkClass.LEGIONNAIRES
                    }
                }

                dhwSolar -> {
                    return if (heatingIncreased) {
                        ChunkClass.SOLAR_COMBINED
                    } else {
                        ChunkClass.SOLAR_DHW
                    }
                }

                dhwIncreased -> {
                    assert(!heatingSolar)
                    return if (heatingIncreased) {
                        ChunkClass.COMBINED
                    } else {
                        ChunkClass.DHW
                    }
                }

                heatingSolar -> {
                    assert(!dhwIncreased)
                    return ChunkClass.SOLAR_HEATING
                }

                heatingIncreased -> {
                    assert(!dhwIncreased)
                    return ChunkClass.HEATING
                }

                else -> {
                    // print(dhw_diff, heating_diff)
                    return ChunkClass.UNKNOWN
                }
            }
        }
}

class DayData(
    timezone: TimeZone,
    timestamps: List<LocalDateTime>,
    data: List<List<Float>>
): Dataset(timezone, timestamps, data) {
//    fun __init__(self, timestamps, full_data):
//        super().__init__(timestamps, full_data)

//        if EXPLORE_UNUSED_DATA:
//            self.explore_unused()
    /*
    fun explore_unused(self):
        speeds = [7, 8]
        plt.figure()
        for i in speeds:
            plt.plot(self.timestamps, self.data[:, i] / 10, label=str(i))
        plt.gca().xaxis.set_major_formatter(TIME_FORMAT)
        plt.legend()
        plt.title("Speeds")
        unused_indices = np.array([i for i in range(self.data.shape[1])
                                   if not (i in self.MAPPING.keys() or i in speeds)])

        plt.figure()
        for i in unused_indices:
            data = self.data[:, i] / 10
            if np.all(np.isclose(data, 0)):
                continue
            plt.plot(self.timestamps, data, label=str(i))
        plt.gca().xaxis.set_major_formatter(TIME_FORMAT)
        plt.legend()

    fun plot(self, axes=None):
        if axes is None:
            plt.figure()
            ax1 = plt.subplot(1, 1, 1)
            ax2 = ax1.twinx()
        else:
            ax1, ax2 = axes
        ax1.plot(self.timestamps, self.consumption, label=f"Electrical power: {self.total_consumption:.2f} kWh",
                 color="red")
        ax1.plot(self.timestamps, self.heating, label=f"Heating power: {self.total_heating:.2f} kWh",
                 color="lightgreen")
        ax1.set_ylabel("kW")
        ax2.plot(self.timestamps, self.cop(), label=f"Mean COP: {self.mean_cop():.2f}")
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper left")
        // plt.suptitle(f"{self.date_str}")
        ax1.xaxis.set_major_formatter(TIME_FORMAT)

        x_data = []
        y_data = []
        labels = []
        properties = []
        for chunk in self.chunks():
            x_data.append(chunk.timestamps[0])
            y_data.append(np.max(chunk.heating) + 0.1)
            label = (f"{chunk.total_consumption:.1f} kWh\n"
                     + f"PF: {chunk.mean_cop():.2f}")
            labels.append(label)
            props = dict()
            try:
                chunk_type = chunk.type
            except ChunkTypeError:
                props.update({"backgroundcolor": "red"})
            else:
                if chunk_type in ChunkClass.legionnaires_types():
                    props.update({"backgroundcolor": "blue"})
                elif chunk_type in ChunkClass.heating_types():
                    props.update({"color": "red"})
                elif chunk_type in ChunkClass.dhw_types():
                    props.update({"color": "blue"})
                if chunk_type in ChunkClass.solar_types():
                    props.update({"bbox": {"edgecolor": "green", "facecolor": "white"}})

            properties.append(props)
        txt_height = 0.07 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        txt_width = datetime.timedelta(hours=1)
        // Get the corrected text positions, then write the text.
        text_positions = get_text_positions(x_data, y_data, txt_width, txt_height)
        text_plotter(x_data, y_data, labels, text_positions, ax1, txt_width, txt_height, properties)

        plt.figure()
        plt.plot(self.timestamps, self.dhw_temp, label="DHW")
        plt.plot(self.timestamps, self.heating_buffer_temp, label="Heating Buffer")
        plt.plot(self.timestamps, self.production_supply, label="Production Flow")
        plt.plot(self.timestamps, self.production_return, label="Production Return")
        plt.plot(self.timestamps, self.brine_supply, label="Brine Return")
        plt.plot(self.timestamps, self.brine_return, label="Brine Return")
        plt.gca().xaxis.set_major_formatter(TIME_FORMAT)
        plt.legend()

        plt.figure()
        plt.plot(self.timestamps, self.outdoor_temp, label="Outdoor")
        plt.gca().xaxis.set_major_formatter(TIME_FORMAT)
        plt.legend()

        plt.show()
        return [ax1, ax2]

 */
}

open class CompositeDataSet(
    timezone: TimeZone,
    val datasets: List<DayData>
): BaseDataset(timezone) { //, metaclass=CompositeMeta):

    override val timestamps: List<LocalDateTime>
        get() = datasets.map { it.timestamps }.flatten() //[ds.timestamps for ds in self.datasets])

    override val data: List<List<Float>>
        get() = datasets.map { it.data }.flatten()
    override val heating: List<Float>
        get() = datasets.map { it.heating }.flatten()
    override val consumption: List<Float>
        get() = datasets.map { it.consumption }.flatten()
    override val cooling: List<Float>
        get() = datasets.map { it.cooling }.flatten()
    override val dhwTemp: List<Float>
        get() = datasets.map { it.dhwTemp }.flatten()
    override val heatingBufferTemp: List<Float>
        get() = datasets.map { it.heatingBufferTemp }.flatten()
    override val coolingBufferTemp: List<Float>
        get() = datasets.map { it.coolingBufferTemp }.flatten()
    override val productionSupply: List<Float>
        get() = datasets.map { it.productionSupply }.flatten()
    override val productionReturn: List<Float>
        get() = datasets.map { it.productionReturn }.flatten()
    override val brineSupply: List<Float>
        get() = datasets.map { it.brineSupply }.flatten()
    override val brineReturn: List<Float>
        get() = datasets.map { it.brineReturn }.flatten()
    override val outdoorTemp: List<Float>
        get() = datasets.map { it.outdoorTemp }.flatten()
//            return np.concatenate([ds.data for ds in self.datasets])
}

class MonthDataSet(
    timezone: TimeZone,
    datasets: List<DayData>
): CompositeDataSet(timezone, datasets){
    fun days(): List<LocalDate>{
        return datasets.map { it.timestamps[0].date }
    }
}
/*
    @functools.cached_property

    fun days(self):
        return np.array([d.timestamps[0].day for d in self.datasets if not d.is_empty])

    fun plot_bar_chart(self):
        plt.figure()
        power_bars = stacked_bar(
            self.days,
            // [d.total_heating for d in self.datasets],
            [d.heating_energy_of_type(ChunkClass.DHW) for d in self.datasets if not d.is_empty],
            [d.heating_energy_of_type(ChunkClass.HEATING) for d in self.datasets if not d.is_empty],
            [d.heating_energy_of_type(ChunkClass.SOLAR_DHW) for d in self.datasets if not d.is_empty],
            [d.heating_energy_of_type(ChunkClass.SOLAR_HEATING) for d in self.datasets if not d.is_empty],
            [d.heating_energy_of_type(ChunkClass.LEGIONNAIRES) for d in self.datasets if not d.is_empty],
            // [d.total_consumption for d in self.datasets],
            )
        colors = [bars.patches[0]._facecolor for bars in power_bars]
        grouped_bar(self.days,
                    // [-d.mean_cop() for d in self.datasets],
                    [-d.mean_cop(ChunkClass.DHW) for d in self.datasets if not d.is_empty],
                    [-d.mean_cop(ChunkClass.HEATING) for d in self.datasets if not d.is_empty],
                    [-d.mean_cop(ChunkClass.SOLAR_DHW) for d in self.datasets if not d.is_empty],
                    [-d.mean_cop(ChunkClass.SOLAR_HEATING) for d in self.datasets if not d.is_empty],
                    [-d.mean_cop(ChunkClass.LEGIONNAIRES) for d in self.datasets if not d.is_empty],
                    // [d.total_consumption for d in self.datasets],
                    colors=colors,
                    )
        plt.legend(["DHW", "Heating", "Solar DHW", "Solar Heating", "Legionnaires"])
        plt.axhline(y=0.0, color="k", linestyle="-")
        plt.show()

 */